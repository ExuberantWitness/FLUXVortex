"""The pairwise total-vorticity gate: the one test that can catch a wrong stretching scheme.

Why this file exists
--------------------
The repo had no test that builds two isolated particles and checks whether the
stretching term is *conservative as a pair*.  It is the physical discriminator
between FLOWVPM's two schemes, and an external audit correctly pointed out that my
axial ``2/5`` identity cannot see the difference (``gamma·(A gamma)`` depends only
on ``sym(A)``, and ``A`` and ``A^T`` share their symmetric part).

For two particles with EQUAL core radius, writing ``d = x_p − x_q`` and
``w = −G(r)/(4 pi r^3)``:

    transposed  S_i = J[m,i] gamma_m   ->   S_p + S_q = 0   EXACTLY
    classic     S_i = J[i,m] gamma_m   ->   S_p + S_q = (w'/r)[ (d·Gamma_p)(d x Gamma_q)
                                                            + (d·Gamma_q)(d x Gamma_p) ] != 0

so total vorticity ``sum(gamma)`` is preserved pairwise only under the transposed
scheme.  That matters here because the deposited LEV clouds are EXACT dipoles with
``sum(gamma_p) = 0`` measured to 1e-17; a non-conservative stretching term would
manufacture spurious monopole vorticity out of a zero-circulation cloud.

Caveat, recorded rather than hidden: the full rVPM update also has ``-3 Z gamma``,
and ``Z`` is scheme-invariant, so neither scheme conserves ``sum(gamma)`` exactly
for the *whole* update.  These tests isolate the ``S`` term, which is where the
schemes differ.
"""

from __future__ import annotations

import pytest
import torch

from pfield_torch_gpu import CudaParticleField

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)

DEVICE = "cuda:0"
SIGMA = 2.0e-4


def _pair(d: torch.Tensor, gp: torch.Tensor, gq: torch.Tensor, sigma_q: float | None = None):
    positions = torch.stack((torch.zeros(3, device=DEVICE, dtype=torch.float64), d))
    gammas = torch.stack((gp, gq))
    field = CudaParticleField(2, device=DEVICE)
    sigmas = torch.tensor(
        [SIGMA, SIGMA if sigma_q is None else sigma_q],
        device=DEVICE,
        dtype=torch.float64,
    )
    field.add_particles(
        positions, gammas, sigmas,
        circul=torch.linalg.vector_norm(gammas, dim=1), birth_step=0,
    )
    return field


def _stretch_pair(d: torch.Tensor, gp: torch.Tensor, gq: torch.Tensor, *, transposed: bool):
    field = _pair(d, gp, gq)
    dt = 1.0e-9
    # rvpm=False isolates dgamma/dt = S, which is the term whose scheme differs.
    field.apply_vortex_stretching(dt, rvpm=False, transposed=transposed)
    delta = field.gamma[:2] - torch.stack((gp, gq))
    return delta / dt


@pytest.mark.parametrize(
    "d, gp, gq",
    [
        ((1.0e-3, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        ((1.0e-3, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        ((0.6e-3, -0.8e-3, 0.3e-3), (0.3, -0.7, 0.5), (0.9, 0.2, -0.4)),
        ((0.0, 1.1e-3, 0.0), (-0.5, 0.5, 0.5), (0.4, -0.9, 0.1)),
    ],
)
def test_transposed_scheme_conserves_total_vorticity_pairwise(d, gp, gq):
    d_t = torch.tensor(d, device=DEVICE, dtype=torch.float64)
    gp_t = torch.tensor(gp, device=DEVICE, dtype=torch.float64)
    gq_t = torch.tensor(gq, device=DEVICE, dtype=torch.float64)

    s = _stretch_pair(d_t, gp_t, gq_t, transposed=True)
    residual = torch.linalg.vector_norm(s.sum(dim=0))
    scale = max(float(torch.linalg.vector_norm(s, dim=1).max().item()), 1.0e-30)
    assert float(residual.item()) <= 1.0e-12 * scale, (
        "transposed scheme must cancel pairwise; "
        f"|S_p+S_q|={float(residual.item()):.3e} scale={scale:.3e}"
    )


@pytest.mark.parametrize(
    "d, gp, gq",
    [
        ((1.0e-3, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        ((0.6e-3, -0.8e-3, 0.3e-3), (0.3, -0.7, 0.5), (0.9, 0.2, -0.4)),
    ],
)
def test_classic_scheme_does_NOT_cancel_pairwise(d, gp, gq):
    """The classic Helmholtz term must fail this, which is what makes the gate real.

    If this ever passes, the two schemes have become indistinguishable in this
    probe and the conservation test above stops meaning anything.
    """

    d_t = torch.tensor(d, device=DEVICE, dtype=torch.float64)
    gp_t = torch.tensor(gp, device=DEVICE, dtype=torch.float64)
    gq_t = torch.tensor(gq, device=DEVICE, dtype=torch.float64)

    s = _stretch_pair(d_t, gp_t, gq_t, transposed=False)
    residual = torch.linalg.vector_norm(s.sum(dim=0))
    scale = max(float(torch.linalg.vector_norm(s, dim=1).max().item()), 1.0e-30)
    assert float(residual.item()) > 1.0e-6 * scale


def test_zero_circulation_dipole_stays_zero_under_the_transposed_scheme():
    """The property that matters for this solver: a dipole cloud must stay a dipole.

    The deposited LEV clouds are exact dipoles (measured sum(gamma) = 1.1e-16 to
    1.7e-17).  Under a non-conservative stretching term they would grow a spurious
    monopole part; under the transposed scheme the pair cancels and the net stays
    zero.  This is the two-particle case with gamma_q = -gamma_p, i.e. the
    physically relevant config, not an abstraction.
    """

    d = torch.tensor((0.7e-3, -0.4e-3, 0.5e-3), device=DEVICE, dtype=torch.float64)
    gp = torch.tensor((0.8, -0.3, 0.6), device=DEVICE, dtype=torch.float64)
    gq = -gp

    for transposed in (True, False):
        field = _pair(d, gp, gq)
        net_before = field.gamma[:2].sum(dim=0)
        assert float(torch.linalg.vector_norm(net_before).item()) < 1.0e-30
        field.apply_vortex_stretching(1.0e-9, rvpm=False, transposed=transposed)
        after = field.gamma[:2]
        net = after.sum(dim=0)
        scale = max(float(torch.linalg.vector_norm(after, dim=1).max().item()), 1.0e-30)
        relative = float(torch.linalg.vector_norm(net).item()) / scale
        if transposed:
            assert relative <= 1.0e-12, f"transposed leaked {relative:.2e} monopole"
        else:
            # The classic scheme DOES leak: recorded here so the gate cannot be
            # mistaken for a no-op.
            assert relative > 1.0e-9, f"classic unexpectedly conserved ({relative:.2e})"
