"""CUDA-only real aerodynamic load packet to Q16 generalized-force gates."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import warp as wp

from fluxvortex.q16_ancf_mesh import make_rectangular_q16_mesh
from fluxvortex.q16_work_conjugate_transfer import Q16SurfaceTransferMap
from fluxvortex.warp_fsi import config
from fluxvortex.warp_fsi.q16_aero_load_packet import (
    Q16CudaAerodynamicLoadPacket,
    Q16CudaResolvedLoadTransfer,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not wp.is_cuda_available(), reason="CUDA required"
)


def _map() -> Q16SurfaceTransferMap:
    mesh = make_rectangular_q16_mesh(
        chordwise_element_count=1,
        spanwise_element_count=1,
        chord=1.7,
        span=0.9,
        thickness=0.12,
    )
    return Q16SurfaceTransferMap(
        mesh=mesh,
        element_indices=np.zeros(5, dtype=np.int64),
        parametric_coordinates=np.array(
            [
                [-0.81, -0.64, 1.0],
                [-0.23, -0.17, 1.0],
                [0.14, 0.28, 1.0],
                [0.63, 0.77, 1.0],
                [0.41, -0.52, 1.0],
            ],
            dtype=np.float64,
        ),
    )


def _state(transfer_map: Q16SurfaceTransferMap) -> np.ndarray:
    rows = transfer_map.mesh.reference_rows.copy()
    rows[:, 2] += 0.015 * rows[:, 0] * rows[:, 1]
    rows[:, 3] += 0.003 * rows[:, 1]
    return np.ascontiguousarray(rows.ravel()[None, :])


def _packet_and_state() -> (
    tuple[Q16SurfaceTransferMap, Q16CudaAerodynamicLoadPacket, wp.array]
):
    transfer_map = _map()
    cuda = Q16CudaResolvedLoadTransfer(transfer_map, device=config.DEVICE)
    q = wp.array(_state(transfer_map), dtype=config.DTYPE, device=config.DEVICE)
    points = wp.to_torch(cuda.surface_transfer.interpolate(q))[0].clone()
    forces = torch.tensor(
        [
            [0.4, -0.2, 0.7],
            [-0.1, 0.6, 0.3],
            [0.8, 0.15, -0.4],
            [-0.35, 0.2, 0.9],
            [0.05, -0.45, 0.25],
        ],
        dtype=torch.float64,
        device=points.device,
    )
    resolved_force = torch.sum(forces, dim=0)
    resolved_moment = torch.sum(torch.linalg.cross(points, forces, dim=1), dim=0)
    packet = Q16CudaAerodynamicLoadPacket.from_tensors(
        point_positions_w=points,
        point_forces_w=forces,
        unresolved_impulse_force_w=torch.zeros_like(resolved_force),
        source_total_force_w=resolved_force,
        source_total_moment_w=resolved_moment,
    )
    return transfer_map, packet, q


def test_resolved_packet_maps_by_exact_q16_transpose_and_virtual_work() -> None:
    transfer_map, packet, q = _packet_and_state()
    transfer = Q16CudaResolvedLoadTransfer(transfer_map, device=config.DEVICE)
    generalized = transfer.map(packet, q)
    direction_np = np.ascontiguousarray(
        np.random.default_rng(20260821).normal(
            scale=0.02, size=(1, transfer_map.structural_dof_count)
        ),
        dtype=np.float64,
    )
    direction = wp.array(direction_np, dtype=config.DTYPE, device=config.DEVICE)
    point_direction = transfer.surface_transfer.interpolate(direction)
    wp.synchronize_device(config.DEVICE)

    structural_work = float(np.sum(direction_np * generalized.numpy()))
    aerodynamic_work = float(
        torch.sum(wp.to_torch(point_direction)[0] * packet.point_forces_w).item()
    )
    assert structural_work == pytest.approx(aerodynamic_work, rel=0.0, abs=2.0e-12)
    np.testing.assert_allclose(
        packet.resolved_force_w.detach().cpu().numpy(),
        np.sum(packet.point_forces_w.detach().cpu().numpy(), axis=0),
        rtol=0.0,
        atol=2.0e-15,
    )


def test_packet_rejects_unresolved_lev_impulse_in_completed_q16_transfer() -> None:
    transfer_map, packet, q = _packet_and_state()
    transfer = Q16CudaResolvedLoadTransfer(transfer_map, device=config.DEVICE)
    hostile = Q16CudaAerodynamicLoadPacket.from_tensors(
        point_positions_w=packet.point_positions_w,
        point_forces_w=packet.point_forces_w,
        unresolved_impulse_force_w=torch.tensor(
            [0.0, 0.0, 1.0e-3], dtype=torch.float64, device="cuda:0"
        ),
        source_total_force_w=packet.resolved_force_w
        + torch.tensor([0.0, 0.0, 1.0e-3], dtype=torch.float64, device="cuda:0"),
        source_total_moment_w=packet.resolved_moment_w,
    )
    with pytest.raises(RuntimeError, match="unresolved LEV impulse"):
        transfer.map(hostile, q)


def test_packet_and_geometry_drift_fail_closed_before_transfer() -> None:
    transfer_map, packet, q = _packet_and_state()
    transfer = Q16CudaResolvedLoadTransfer(transfer_map, device=config.DEVICE)

    packet.point_forces_w[0, 0] += 1.0
    with pytest.raises(RuntimeError, match="packet content drift"):
        transfer.map(packet, q)

    _, clean, clean_q = _packet_and_state()
    clean.point_positions_w[2, 1] += 2.0e-4
    resealed = Q16CudaAerodynamicLoadPacket.from_tensors(
        point_positions_w=clean.point_positions_w,
        point_forces_w=clean.point_forces_w,
        unresolved_impulse_force_w=clean.unresolved_impulse_force_w,
        source_total_force_w=clean.source_total_force_w,
        source_total_moment_w=torch.sum(
            torch.linalg.cross(clean.point_positions_w, clean.point_forces_w, dim=1),
            dim=0,
        ),
    )
    with pytest.raises(RuntimeError, match="aero/Q16 geometry mismatch"):
        transfer.map(resealed, clean_q)


def test_packet_rejects_host_or_float32_scientific_arrays() -> None:
    cuda = torch.device("cuda:0")
    good = torch.zeros((1, 3), dtype=torch.float64, device=cuda)
    with pytest.raises(ValueError, match="CUDA"):
        Q16CudaAerodynamicLoadPacket.from_tensors(
            point_positions_w=torch.zeros((1, 3), dtype=torch.float64),
            point_forces_w=good,
            unresolved_impulse_force_w=torch.zeros(3, dtype=torch.float64, device=cuda),
            source_total_force_w=torch.zeros(3, dtype=torch.float64, device=cuda),
            source_total_moment_w=torch.zeros(3, dtype=torch.float64, device=cuda),
        )
    with pytest.raises(TypeError, match="float64"):
        Q16CudaAerodynamicLoadPacket.from_tensors(
            point_positions_w=good,
            point_forces_w=torch.zeros((1, 3), dtype=torch.float32, device=cuda),
            unresolved_impulse_force_w=torch.zeros(3, dtype=torch.float64, device=cuda),
            source_total_force_w=torch.zeros(3, dtype=torch.float64, device=cuda),
            source_total_moment_w=torch.zeros(3, dtype=torch.float64, device=cuda),
        )


def test_solver_extraction_requires_mandatory_lev_tev_free_wake_owner() -> None:
    class ArrayCarrier:
        pass

    hostile = ArrayCarrier()
    with pytest.raises(RuntimeError, match="configuration owner"):
        Q16CudaAerodynamicLoadPacket.from_solver(hostile)
