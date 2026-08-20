"""CUDA-resident particle field regression and attack tests."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from fluxvortex.rvpm_reference import direct_gaussian_erf_velocity_jacobian
from pfield_torch_gpu import CudaParticleField, TYPE_FREE

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _fixture() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = torch.device("cuda:0")
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [0.7, -0.2, 0.1], [-0.4, 0.3, 0.8]],
        device=device,
        dtype=torch.float64,
    )
    gamma = torch.tensor(
        [[0.2, -0.5, 0.7], [0.6, 0.1, -0.3], [-0.4, 0.9, 0.2]],
        device=device,
        dtype=torch.float64,
    )
    sigma = torch.tensor([0.2, 0.35, 0.4], device=device, dtype=torch.float64)
    return pos, gamma, sigma


def test_cuda_resident_velocity_matches_independent_oracle() -> None:
    pos, gamma, sigma = _fixture()
    field = CudaParticleField(capacity=8)
    field.add_particles(pos, gamma, sigma)
    actual = field.velocity_self_cuda()
    oracle = direct_gaussian_erf_velocity_jacobian(
        pos.cpu().numpy(), gamma.cpu().numpy(), sigma.cpu().numpy()
    ).velocity
    assert actual.is_cuda and field.positions_cuda.is_cuda
    np.testing.assert_allclose(
        actual.detach().cpu().numpy(), oracle, rtol=3.0e-13, atol=3.0e-13
    )


def test_particle_field_rejects_host_and_mixed_dtype_inputs() -> None:
    pos, gamma, sigma = _fixture()
    field = CudaParticleField(capacity=8)
    with pytest.raises(ValueError, match="must be on cuda"):
        field.add_particles(pos.cpu(), gamma, sigma)
    with pytest.raises(TypeError, match="float64"):
        field.add_particles(pos, gamma.float(), sigma)


def test_wrk3_advance_stays_on_cuda_and_promotes_fresh() -> None:
    pos, gamma, sigma = _fixture()
    field = CudaParticleField(capacity=8)
    field.add_particles(pos, gamma, sigma)
    initial = field.positions_cuda.clone()
    constant = torch.tensor([0.2, -0.1, 0.05], device="cuda:0", dtype=torch.float64)
    field.advance_wrk3(0.01, lambda points: constant.expand_as(points))
    assert field.positions_cuda.is_cuda
    assert not torch.equal(field.positions_cuda, initial)
    assert bool(torch.all(field.types_cuda == TYPE_FREE).item())


def test_ring_shedding_is_cuda_and_capacity_bounded() -> None:
    field = CudaParticleField(capacity=4)
    ring = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]],
        device="cuda:0",
        dtype=torch.float64,
    )
    strength = torch.tensor([0.4], device="cuda:0", dtype=torch.float64)
    assert (
        field.add_ring_particles(ring, strength, sigma_factor=17.5, birth_step=3) == 4
    )
    assert field.n == 4
    with pytest.raises(OverflowError, match="capacity exceeded"):
        field.add_ring_particles(ring, strength, sigma_factor=17.5, birth_step=4)
