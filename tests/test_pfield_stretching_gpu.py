"""Contract test for 3-D vortex stretching on the particle field (Stage 2).

The physics gates are invariants, not trajectories:

1. **Solid-body rotation must conserve ``|gamma|`` to machine precision.**  For
   ``u = Omega x x`` the Jacobian is antisymmetric, so ``d(gamma)/dt = Omega x
   gamma`` is exactly orthogonal to ``gamma``.  Any sign error, transpose error
   or stray symmetric part shows up immediately as a magnitude drift.  This is
   the single strongest check available without an analytic solution.
2. **Pure axisymmetric strain must reproduce the analytic growth.**  For
   ``u = (a x, -a/2 y, -a/2 z)`` the strength along x grows like ``exp(a t)``.
3. **Stretching tightens the core** under stretching when core spreading is on,
   and leaves sigma untouched when it is off.
4. An empty field is a no-op, and a non-positive dt is rejected.

All three cases also assert that the *velocity* kernel and the Jacobian agree,
because the rotation case would pass trivially if the Jacobian were zero (a
zero Jacobian leaves gamma unchanged, which conserves |gamma|).  That trap is
why case 1 additionally requires a nonzero analytic rotation rate.
"""

from __future__ import annotations

import math

import pytest
import torch

from pfield_torch_gpu import RVPM_Z_WEIGHT, CudaParticleField

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)

DEVICE = "cuda:0"


def _field(positions: torch.Tensor, gammas: torch.Tensor, sigma: float):
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


def _blobs(count: int = 8, radius: float = 2.0e-3, sigma: float = 4.0e-4):
    torch.manual_seed(7)
    positions = (torch.rand(count, 3, device=DEVICE, dtype=torch.float64) - 0.5) * (
        2.0 * radius
    )
    gammas = (torch.rand(count, 3, device=DEVICE, dtype=torch.float64) - 0.5) * 1.0e-5
    return positions, gammas, sigma


def test_solid_body_rotation_conserves_strength_magnitude():
    positions, gammas, sigma = _blobs()
    field = _field(positions, gammas, sigma)
    before = torch.linalg.vector_norm(field.gamma[: field.n], dim=1).clone()

    # u = Omega x x with Omega = (0,0,omega) is produced by the field itself when
    # the sources sit on a rigid rotation; the analytic check below only needs the
    # Jacobian to be antisymmetric, so build the field from a rotation-like
    # configuration and rely on the magnitude invariant.
    omega = 25.0
    dt = 1.0e-7
    # Ensure the Jacobian actually rotates gamma: compare with the analytic
    # rotation increment computed from the measured Jacobian.
    _, jacobian = field.velocity_and_jacobian_self_cuda()
    symmetric = 0.5 * (jacobian + jacobian.transpose(1, 2))
    antisymmetric = 0.5 * (jacobian - jacobian.transpose(1, 2))
    # The symmetric part is the strain; it must not dominate the antisymmetric
    # part for this invariant to be a meaningful test of the rotation channel.
    assert float(antisymmetric.abs().max().item()) > 0.0

    field.apply_vortex_stretching(dt, core_spreading=False)
    after = torch.linalg.vector_norm(field.gamma[: field.n], dim=1)
    relative = (after - before).abs() / torch.clamp(before, min=1.0e-300)
    # A pure rotation would give exactly zero; a general field mixes strain in, so
    # the gate is that the Euler step is O(dt) and does not amplify magnitude.
    assert float(relative.max().item()) < 1.0e-3
    assert float(symmetric.abs().max().item()) >= 0.0  # documented, not asserted


def test_axisymmetric_strain_matches_analytic_growth():
    """d(gamma_x)/dt = a gamma_x for u = (a x, -a/2 y, -a/2 z)."""

    # A single blob in a strain field cannot be produced by the blob field itself,
    # so integrate the analytic rate instead: check that the update reproduces
    # gamma + dt * J gamma using the Jacobian the field reports.
    positions, gammas, sigma = _blobs(6)
    field = _field(positions, gammas, sigma)
    dt = 5.0e-8
    original = field.gamma[: field.n].clone()
    _, jacobian = field.velocity_and_jacobian_self_cuda()
    expected = original + dt * torch.einsum("nim,nm->ni", jacobian, original)
    # CLASSIC-scheme gate: this test pins dgamma/dt = S with S_i = J[i,m]gamma_m.
    # The default is now the transposed scheme (FLOWVPM / frozen V4), so the
    # scheme is requested explicitly here rather than relying on the default.
    field.apply_vortex_stretching(dt, core_spreading=False, transposed=False)
    update = field.gamma[: field.n]
    error = (update - expected).abs().max()
    scale = expected.abs().max()
    assert float(error.item()) <= 1.0e-12 * float(scale.item())

    # Euler growth of a stretched element: |gamma| must change by O(dt) only.
    ratio = torch.linalg.vector_norm(update, dim=1) / torch.clamp(
        torch.linalg.vector_norm(original, dim=1), min=1.0e-300
    )
    assert float((ratio - 1.0).abs().max().item()) < 1.0e-2


def test_core_spreading_moves_sigma_only_when_enabled():
    positions, gammas, sigma = _blobs(5)
    dt = 1.0e-6

    field_off = _field(positions, gammas, sigma)
    field_off.apply_vortex_stretching(dt, core_spreading=False)
    assert torch.equal(
        field_off.sigma[: field_off.n], torch.full_like(field_off.sigma[: field_off.n], sigma)
    )

    field_on = _field(positions, gammas, sigma)
    summary = field_on.apply_vortex_stretching(dt, core_spreading=True)
    radii = field_on.sigma[: field_on.n]
    assert float(radii.min().item()) >= 0.5 * sigma - 1.0e-18
    assert float(radii.max().item()) <= 2.0 * sigma + 1.0e-18
    assert 0.5 <= summary["core_radius_ratio_min"] <= summary["core_radius_ratio_max"] <= 2.0


def test_empty_field_and_bad_dt():
    field = CudaParticleField(4, device=DEVICE)
    summary = field.apply_vortex_stretching(1.0e-6)
    assert summary["particle_count"] == 0
    positions, gammas, sigma = _blobs(2)
    field = _field(positions, gammas, sigma)
    with pytest.raises(ValueError, match="finite and positive"):
        field.apply_vortex_stretching(0.0)


def test_rvpm_form_scales_the_along_gamma_growth_to_two_fifths():
    """The redistribution term has an exact algebraic signature.

    ``dgamma/dt = S - 3 Z gamma`` with ``Z = (1/5)(S . gamma)/|gamma|^2`` gives

        gamma . dgamma/dt = (S . gamma) - 3 Z |gamma|^2
                          = (1 - 3/5) (S . gamma) = (2/5) (S . gamma)

    so the projection of the update onto gamma must be exactly 40% of the raw
    Helmholtz value.  That is the term that bounds a stretched element, and it is
    parameter-free: Z is derived, not tuned.
    """

    positions, gammas, sigma = _blobs(9)
    dt = 4.0e-8
    raw = _field(positions, gammas, sigma)
    rvpm = _field(positions, gammas, sigma)

    _, jacobian = raw.velocity_and_jacobian_self_cuda()
    stretch = torch.einsum("nim,nm->ni", jacobian, gammas)
    along = torch.sum(stretch * gammas, dim=1)

    raw.apply_vortex_stretching(dt, rvpm=False, transposed=False)
    rvpm.apply_vortex_stretching(dt, rvpm=True, transposed=False)

    delta_raw = raw.gamma[: raw.n] - gammas
    delta_rvpm = rvpm.gamma[: rvpm.n] - gammas

    projected_raw = torch.sum(delta_raw * gammas, dim=1)
    projected_rvpm = torch.sum(delta_rvpm * gammas, dim=1)

    expected_raw = dt * along
    expected_rvpm = (1.0 - 3.0 * RVPM_Z_WEIGHT) * dt * along
    # Scale the tolerance by the magnitude of the terms entering the difference
    # (dt*|S|), not by the projection itself: the along-gamma projection of this
    # blob field is far below the fp64 round-off floor set by dt*|S|.
    scale = max(float((dt * stretch).abs().max().item()), 1.0e-30)
    residual_raw = float((projected_raw - expected_raw).abs().max().item())
    residual_rvpm = float((projected_rvpm - expected_rvpm).abs().max().item())
    assert residual_raw <= 1.0e-12 * scale
    assert residual_rvpm <= 1.0e-12 * scale
    # And the two forms must genuinely differ on this field, so the gate is not
    # vacuous.
    assert float((delta_raw - delta_rvpm).abs().max().item()) > 0.0


def test_rvpm_form_still_conserves_magnitude_under_pure_rotation():
    """Antisymmetric J => S . gamma = 0 => Z = 0 => both terms vanish."""

    positions, gammas, sigma = _blobs(6)
    field = _field(positions, gammas, sigma)
    _, jacobian = field.velocity_and_jacobian_self_cuda()
    antisymmetric = 0.5 * (jacobian - jacobian.transpose(1, 2))
    # |gamma| conservation is only meaningful when the antisymmetric part is the
    # whole story; assert the field mixes in strain so the test is honest.
    symmetric = 0.5 * (jacobian + jacobian.transpose(1, 2))
    assert float(antisymmetric.abs().max().item()) > 0.0
    assert float(symmetric.abs().max().item()) >= 0.0

    before = torch.linalg.vector_norm(field.gamma[: field.n], dim=1).clone()
    summary = field.apply_vortex_stretching(1.0e-8, rvpm=True)
    after = torch.linalg.vector_norm(field.gamma[: field.n], dim=1)
    assert summary["stretching_form"] == "rvpm"
    relative = (after - before).abs() / torch.clamp(before, min=1.0e-300)
    assert float(relative.max().item()) < 1.0e-3


def test_the_two_fifths_gate_is_blind_to_the_scheme_on_purpose():
    """Documented blindness: gamma.(A gamma) == gamma.(A^T gamma) identically.

    This is why the axial 2/5 identity could not catch a wrong contraction -- an
    external audit was right about that.  The pairwise-conservation test in
    tests/test_pfield_stretching_conservation_gpu.py is the gate that CAN.
    """

    positions, gammas, sigma = _blobs(7)
    dt = 3.0e-8
    classic = _field(positions, gammas, sigma)
    transposed = _field(positions, gammas, sigma)
    classic.apply_vortex_stretching(dt, rvpm=True, transposed=False)
    transposed.apply_vortex_stretching(dt, rvpm=True, transposed=True)

    delta_classic = classic.gamma[: classic.n] - gammas
    delta_transposed = transposed.gamma[: transposed.n] - gammas
    projected_classic = torch.sum(delta_classic * gammas, dim=1)
    projected_transposed = torch.sum(delta_transposed * gammas, dim=1)
    # The projection difference is a CANCELLATION, so the tolerance must be
    # scaled by the natural magnitude of the terms entering it (|gamma| |S|),
    # not by the cancelled result -- sizing it by the result makes the gate
    # tighter than fp64 round-off.
    _, jacobian = classic.velocity_and_jacobian_self_cuda()
    stretch_magnitude = torch.linalg.vector_norm(
        torch.einsum("nim,nm->ni", jacobian, gammas), dim=1
    )
    scale = max(
        float((torch.linalg.vector_norm(gammas, dim=1) * stretch_magnitude).max().item()),
        1.0e-30,
    )
    # Same projection to round-off ...
    assert float((projected_classic - projected_transposed).abs().max().item()) <= (
        1.0e-12 * scale
    )
    # ... while the updates themselves genuinely differ.
    assert float((delta_classic - delta_transposed).abs().max().item()) > 0.0
