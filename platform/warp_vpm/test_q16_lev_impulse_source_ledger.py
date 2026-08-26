"""CUDA source ownership and spanwise impulse-ledger gates for Q16 FSI."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from bing_joint_ptera import JointConfig
from bing_joint_ptera_gpu import CudaJointLEVTEVSolver
from fluxvortex.warp_fsi.q16_aero_load_packet import Q16CudaAerodynamicLoadPacket
from pfield_torch_gpu import CudaParticleField
from q16_incremental_ptera_owner import (
    Q16CudaIncrementalAeroSession,
    Q16IncrementalAeroLifecycleError,
)
from test_ptera_gpu_active_lev import _problem


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _rings() -> tuple[torch.Tensor, torch.Tensor]:
    rings = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.2, 0.8, 0.1],
                [-0.1, 0.6, -0.2],
            ],
            [
                [2.0, -0.4, 0.3],
                [2.7, -0.2, 0.1],
                [2.9, 0.5, 0.4],
                [1.8, 0.7, 0.2],
            ],
        ],
        dtype=torch.float64,
        device="cuda:0",
    )
    strengths = torch.tensor([0.4, -0.25], dtype=torch.float64, device="cuda:0")
    return rings, strengths


@pytest.mark.parametrize("reverse", [False, True])
def test_ring_particles_match_independent_oriented_edge_oracle(
    reverse: bool,
) -> None:
    rings, strengths = _rings()
    field = CudaParticleField(capacity=16)
    count = field.add_ring_particles(
        rings,
        strengths,
        sigma_factor=17.5,
        birth_step=3,
        reverse=reverse,
    )
    step = -1 if reverse else 1
    rings_host = rings.cpu().numpy()
    strengths_host = strengths.cpu().numpy()
    positions: list[np.ndarray] = []
    gammas: list[np.ndarray] = []
    sigmas: list[float] = []
    circulations: list[float] = []
    source_strips: list[int] = []
    for strip, (ring, strength) in enumerate(
        zip(rings_host, strengths_host, strict=True)
    ):
        for leg in range(4):
            origin = ring[leg]
            destination = ring[(leg + step) % 4]
            vector = destination - origin
            length = float(np.linalg.norm(vector))
            positions.append(0.5 * (origin + destination))
            gammas.append(vector * strength)
            sigmas.append(length / 17.5)
            circulations.append(float(strength))
            source_strips.append(strip)

    snapshot = field.snapshot_numpy()
    assert count == field.n == 8
    np.testing.assert_array_equal(snapshot["positions"], np.asarray(positions))
    np.testing.assert_array_equal(snapshot["gamma"], np.asarray(gammas))
    np.testing.assert_allclose(snapshot["sigma"], sigmas, rtol=3.0e-16, atol=0.0)
    np.testing.assert_array_equal(snapshot["circul"], circulations)
    np.testing.assert_array_equal(snapshot["source_strip"], source_strips)


def test_source_strip_survives_compaction_and_invalid_input_is_atomic() -> None:
    rings, strengths = _rings()
    field = CudaParticleField(capacity=16)
    field.add_ring_particles(
        rings,
        strengths,
        sigma_factor=17.5,
        birth_step=3,
    )
    before = field.snapshot_numpy()
    remove = torch.tensor(
        [False, True, False, True, True, False, False, True],
        dtype=torch.bool,
        device="cuda:0",
    )
    field.remove_mask(remove)
    after = field.snapshot_numpy()
    keep = ~remove.cpu().numpy()
    for name in before:
        np.testing.assert_array_equal(after[name], before[name][keep])

    pos = rings.reshape(-1, 3)[:2]
    gamma = torch.ones_like(pos)
    sigma = torch.ones(2, dtype=torch.float64, device="cuda:0")
    state = field.snapshot_numpy()
    invalid = (
        torch.tensor([0, 1], dtype=torch.int64),
        torch.tensor([0, 1], dtype=torch.int32, device="cuda:0"),
        torch.tensor([[0, 1]], dtype=torch.int64, device="cuda:0"),
        torch.tensor([0, -2], dtype=torch.int64, device="cuda:0"),
    )
    for source_strip in invalid:
        with pytest.raises((TypeError, ValueError)):
            field.add_particles(
                pos,
                gamma,
                sigma,
                source_strip=source_strip,
            )
        assert field.n == len(after["source_strip"])
        for name, expected in state.items():
            np.testing.assert_array_equal(field.snapshot_numpy()[name], expected)


def _solver(*, steps: int) -> CudaJointLEVTEVSolver:
    solver = CudaJointLEVTEVSolver(
        _problem(steps),
        JointConfig(
            enable_lev=True,
            joint_tev=True,
            lesp_crit=0.001,
            lev_start_step=0,
            particle_capacity=256,
        ),
        device="cuda:0",
    )
    solver._prescribed_wake = False
    return solver


def _independent_strip_impulse(solver: CudaJointLEVTEVSolver) -> torch.Tensor:
    _, span_count, chord_count = solver._panel_grid()
    device = solver.cuda_device
    rotation = solver._v5m_gp_to_scientific_rotation_cuda
    translation = solver._v5m_gp_to_scientific_translation_cuda
    positions = solver.lev_pf.positions_cuda
    gammas = solver.lev_pf.gammas_cuda
    sources = solver.lev_pf.source_strips_cuda
    positions_w = torch.einsum("ij,nj->ni", rotation, positions) + translation
    gammas_w = torch.einsum("ij,nj->ni", rotation, gammas)
    particle_terms = (
        0.5
        * float(solver.current_operating_point.rho)
        * (
            torch.linalg.cross(positions_w, gammas_w, dim=1)
            + torch.linalg.cross(translation.expand_as(gammas_w), gammas_w, dim=1)
        )
    )
    free = torch.zeros((span_count, 3), dtype=torch.float64, device=device)
    for strip in range(span_count):
        free[strip] = torch.sum(particle_terms[sources == strip], dim=0)
    normals = torch.as_tensor(
        np.ascontiguousarray(solver.stackUnitNormals_GP1),
        dtype=torch.float64,
        device=device,
    )
    normals_w = torch.einsum("ij,nj->ni", rotation, normals)
    area = torch.as_tensor(
        np.ascontiguousarray(solver.panel_areas),
        dtype=torch.float64,
        device=device,
    )
    bound = (
        float(solver.current_operating_point.rho)
        * solver._cuda_bound_strengths.unsqueeze(1)
        * area.unsqueeze(1)
        * normals_w
    ).reshape(chord_count, span_count, 3)
    return free + torch.sum(bound, dim=0)


def test_real_two_step_strip_impulse_closes_frozen_hirato_time_layer_result() -> None:
    solver = _solver(steps=2)
    session = Q16CudaIncrementalAeroSession.begin(solver)
    session.advance_one_step()
    previous_strip_impulse = _independent_strip_impulse(solver).clone()
    session.advance_one_step()
    current_strip_impulse = _independent_strip_impulse(solver).clone()
    expected_strip_force = -(current_strip_impulse - previous_strip_impulse) / float(
        solver.delta_time
    )
    session.finalize()
    packet = Q16CudaAerodynamicLoadPacket.from_solver(solver)
    expected_impulse = torch.tensor(
        [
            float.fromhex("-0x1.ca31cd43dc9b5p+2"),
            float.fromhex("-0x1.3c3d300000001p-40"),
            float.fromhex("-0x1.34e691005ae73p+3"),
        ],
        dtype=torch.float64,
        device="cuda:0",
    )
    assert solver.lev_pf.n == 24
    assert packet.packet_sha256 == (
        "5617ab81d33875f84eef9e182d62903c92db5477e1355cae8e1523e4e688949e"
    )
    assert max(diagnostic["kelvin_eq9_max_abs"] for diagnostic in solver.diag) == 0.0
    torch.testing.assert_close(
        packet.unresolved_impulse_force_w, expected_impulse, rtol=0.0, atol=0.0
    )

    strip_force = solver._q16_impulse_strip_force_w
    endpoints = solver._q16_impulse_strip_le_endpoints_w
    assert strip_force.shape == (3, 3)
    assert endpoints.shape == (3, 2, 3)
    assert strip_force.device.type == endpoints.device.type == "cuda"
    assert strip_force.dtype is endpoints.dtype is torch.float64
    assert bool(torch.isfinite(strip_force).all().item())
    assert bool(torch.isfinite(endpoints).all().item())
    torch.testing.assert_close(strip_force, expected_strip_force, rtol=0.0, atol=0.0)
    closure_atol = (
        16.0
        * np.finfo(np.float64).eps
        * max(1.0, float(torch.max(torch.abs(packet.unresolved_impulse_force_w))))
    )
    torch.testing.assert_close(
        torch.sum(strip_force, dim=0),
        packet.unresolved_impulse_force_w,
        rtol=0.0,
        atol=closure_atol,
    )

    source_strip = solver.lev_pf.source_strips_cuda
    assert source_strip.dtype is torch.int64 and source_strip.is_cuda
    assert int(torch.min(source_strip).item()) == 0
    assert int(torch.max(source_strip).item()) == 2
    assert set(source_strip.cpu().tolist()) == {0, 1, 2}

    leading_gp, _ = solver._station_le_te_points()
    leading = torch.as_tensor(leading_gp, dtype=torch.float64, device="cuda:0")
    leading_w = solver._v5m_gp_points_to_scientific_cuda(leading)
    expected_endpoints = torch.stack((leading_w[:-1], leading_w[1:]), dim=1)
    torch.testing.assert_close(endpoints, expected_endpoints, rtol=0.0, atol=0.0)


def test_incremental_owner_detects_source_identity_drift_and_clean_retry() -> None:
    session = Q16CudaIncrementalAeroSession.begin(_solver(steps=3))
    session.advance_one_step()
    assert session.solver.lev_pf.n > 0

    hostile = session.fork()
    hostile.solver.lev_pf.source_strip[0] += 1
    with pytest.raises(Q16IncrementalAeroLifecycleError, match="state drift"):
        hostile.advance_one_step()

    clean_a = session.fork()
    clean_b = session.fork()
    receipt_a = clean_a.advance_one_step()
    receipt_b = clean_b.advance_one_step()
    assert receipt_a == receipt_b
    assert torch.equal(
        clean_a.solver.lev_pf.source_strips_cuda,
        clean_b.solver.lev_pf.source_strips_cuda,
    )
