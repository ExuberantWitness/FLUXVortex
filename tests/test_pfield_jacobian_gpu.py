"""Contract test for the analytic blob Jacobian (the piece vortex stretching needs).

Why this test exists
--------------------
Stage 2 (3-D vortex stretching + core spreading) needs ``J[i,m] = du_i/dx_m`` at
the live particle positions.  ``pfield_torch_gpu.py`` had only the velocity kernel,
so ``_pfield_jacobian_partials`` was written as a same-loop mirror of it.  The
failure modes that must be caught are:

* a wrong analytic derivative (silently changes the stretching rate);
* a wrong index convention -- this file stores ``J[i,m]``, so stretching is
  ``einsum("nim,nm->ni")``; ``src/fluxvortex/particles.py:149`` stores the same
  layout but contracts the wrong index, which a prior audit already flagged;
* a wrong far-field early-out (``rho > 12``): the gradient must stay valid there
  even though ``G' `` is dropped;
* a sign error in the antisymmetric term, which is trace-free and therefore
  invisible to the trace identity alone.

Gates:
1. ``tr(J) == 0`` to machine precision -- analytic identity, independent of
   discretisation, so it catches a whole class of sign/arrangement errors;
2. ``J`` against central finite differences of ``velocity_at_cuda``;
3. ``J @ gamma`` against a directional finite difference along ``gamma`` -- this
   is exactly the stretching rate, so it pins the convention end to end;
4. both gates hold across the ``rho`` regimes (near, peak, and the rho>12
   early-out branch).
"""

from __future__ import annotations

import pytest
import torch

from pfield_torch_gpu import CudaParticleField

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)

DEVICE = "cuda:0"


def _blob_field(sigma: float, positions: torch.Tensor, gammas: torch.Tensor):
    field = CudaParticleField(max(positions.shape[0], 1), device=DEVICE)
    count = positions.shape[0]
    field.add_particles(
        positions,
        gammas,
        torch.full((count,), sigma, device=DEVICE, dtype=torch.float64),
        circul=torch.linalg.vector_norm(gammas, dim=1),
        birth_step=0,
    )
    return field


def _cluster(sigma: float, span: float, count: int = 12) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(20261121)
    positions = (torch.rand(count, 3, device=DEVICE, dtype=torch.float64) - 0.5) * span
    gammas = (torch.rand(count, 3, device=DEVICE, dtype=torch.float64) - 0.5) * 1.0e-3
    return positions, gammas


@pytest.mark.parametrize("span_over_sigma", [0.5, 3.0, 40.0])
def test_trace_vanishes_identically(span_over_sigma):
    """div(u) = 0 for a Gaussian blob field, to machine precision."""

    sigma = 1.0e-3
    positions, gammas = _cluster(sigma, span_over_sigma * sigma)
    field = _blob_field(sigma, positions, gammas)
    targets = positions + torch.tensor([0.7, -0.3, 0.4], device=DEVICE, dtype=torch.float64) * span_over_sigma * sigma
    _, jacobian = field.velocity_and_jacobian_self_cuda(targets)
    trace = jacobian[:, 0, 0] + jacobian[:, 1, 1] + jacobian[:, 2, 2]
    scale = torch.linalg.vector_norm(jacobian.reshape(-1, 9), dim=1).max()
    assert float(trace.abs().max().item()) <= 1.0e-12 * max(float(scale.item()), 1.0e-30)


@pytest.mark.parametrize("span_over_sigma", [0.5, 3.0, 40.0])
def test_jacobian_matches_central_difference(span_over_sigma):
    """J[i,m] must equal d u_i / d x_m by central differences."""

    sigma = 1.0e-3
    positions, gammas = _cluster(sigma, span_over_sigma * sigma)
    field = _blob_field(sigma, positions, gammas)
    targets = positions + torch.tensor([0.55, 0.2, -0.35], device=DEVICE, dtype=torch.float64) * span_over_sigma * sigma
    _, jacobian = field.velocity_and_jacobian_self_cuda(targets)

    # Finite-difference step must resolve the kernel: h = sigma/500 is small
    # compared with sigma but far above fp64 round-off at these magnitudes.
    step = sigma / 500.0
    numerical = torch.zeros_like(jacobian)
    for m in range(3):
        offset = torch.zeros(3, device=DEVICE, dtype=torch.float64)
        offset[m] = step
        plus = field.velocity_at_cuda(targets + offset)
        minus = field.velocity_at_cuda(targets - offset)
        numerical[:, :, m] = (plus - minus) / (2.0 * step)

    error = torch.linalg.vector_norm((jacobian - numerical).reshape(-1, 9), dim=1)
    reference = torch.linalg.vector_norm(numerical.reshape(-1, 9), dim=1)
    relative = error / torch.clamp(reference, min=1.0e-300)
    assert float(relative.max().item()) < 2.0e-6


@pytest.mark.parametrize("span_over_sigma", [0.5, 3.0, 40.0])
def test_stretching_rate_contracts_over_the_second_index(span_over_sigma):
    """d(gamma)/dt = J . gamma must match a directional finite difference.

    This is the convention gate: contracting the wrong index produces a
    numerically different vector here (the Jacobian is not symmetric), so the
    test fails loudly on the classic bug.
    """

    sigma = 1.0e-3
    positions, gammas = _cluster(sigma, span_over_sigma * sigma)
    field = _blob_field(sigma, positions, gammas)
    targets = positions + torch.tensor([0.35, -0.5, 0.15], device=DEVICE, dtype=torch.float64) * span_over_sigma * sigma
    _, jacobian = field.velocity_and_jacobian_self_cuda(targets)

    stretch = torch.einsum("nim,nm->ni", jacobian, gammas)
    wrong_contraction = torch.einsum("nmi,nm->ni", jacobian, gammas)

    step = 1.0e-3 * span_over_sigma * sigma
    moved = targets + step * gammas
    numerical = (field.velocity_at_cuda(moved) - field.velocity_at_cuda(targets)) / step
    relative = torch.linalg.vector_norm(stretch - numerical, dim=1) / torch.clamp(
        torch.linalg.vector_norm(numerical, dim=1), min=1.0e-300
    )
    assert float(relative.max().item()) < 1.0e-4
    # And the two contractions must be genuinely different numbers, so the gate
    # above is not vacuous for this field.
    difference = (stretch - wrong_contraction).abs().max()
    assert float(difference.item()) > 0.0


def test_empty_field_returns_zeros():
    field = CudaParticleField(4, device=DEVICE)
    velocity, jacobian = field.velocity_and_jacobian_self_cuda()
    assert velocity.shape == (0, 3)
    assert jacobian.shape == (0, 3, 3)
