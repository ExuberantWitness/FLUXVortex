"""Contract tests for the physically weighted load metrics (CPU only, no GPU needed).

The point of these metrics is that they must NOT behave like a raw Euclidean norm of a
mixed-unit vector.  The decisive test is therefore a *units* test: scale one DOF block
by a large factor and show the flexibility norm is (nearly) unchanged while the raw L2
norm moves a lot.  That is exactly the failure mode that produced the 0.338 number.
"""

from __future__ import annotations

import math

import pytest
import torch

from fluxvortex.warp_fsi.q16_load_metrics import (
    flexibility_energy_norm,
    relative_flexibility_gap,
)


def _spd(dof: int, seed: int = 5) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    a = torch.randn(dof, dof, generator=generator, dtype=torch.float64)
    # K = A A^T + dof*I is SPD with a bounded condition number.
    return a @ a.transpose(0, 1) + dof * torch.eye(dof, dtype=torch.float64)


def test_norm_is_homogeneous_of_degree_one():
    """sqrt(r^T K^-1 r) scales linearly with the load, as an energy^1/2 must."""

    dof = 12
    k = _spd(dof)
    r = torch.randn(dof, dtype=torch.float64)
    base = flexibility_energy_norm(r, k)
    for scale in (0.5, 2.0, 7.0):
        assert math.isclose(
            flexibility_energy_norm(scale * r, k), scale * base, rel_tol=1e-12
        )


def test_zero_residual_is_zero():
    assert flexibility_energy_norm(torch.zeros(4, dtype=torch.float64), _spd(4)) == 0.0


def test_norm_respects_single_dof_stiffness():
    """One DOF with stiffness 100 and load 10 must give sqrt(10^2/100) = 1."""

    k = torch.diag(torch.tensor([100.0], dtype=torch.float64))
    r = torch.tensor([10.0], dtype=torch.float64)
    assert math.isclose(flexibility_energy_norm(r, k), 1.0, rel_tol=1e-12)


def test_units_test_the_flexibility_norm_is_insensitive_to_dof_scaling():
    """The whole reason this module exists.

    A change of DOF units ``q' = c q`` transforms the conjugate load and the stiffness
    as ``r' = r / c`` and ``K' = K / (c c^T)``, which leaves ``r . u`` (and therefore
    the flexibility norm) EXACTLY invariant.  A raw L2 norm is not invariant at all:
    here the second block is expressed 1000x smaller in its own units, which moves the
    raw norm by ~3 orders of magnitude while the physical content is identical.
    """

    stiffness = torch.diag(
        torch.tensor([1.0] * 4 + [1.0e6] * 4, dtype=torch.float64)
    )
    physical = torch.tensor([1.0, 2.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0], dtype=torch.float64)
    # c = 1e-3 on the second block: r' = r / c = 1000 r, K' = K / c^2 = 1e6 K.
    rescaled = physical.clone()
    rescaled[4:] = rescaled[4:] / 1.0e-3
    rescaled_stiffness = stiffness.clone()
    rescaled_stiffness[4:, 4:] = rescaled_stiffness[4:, 4:] / (1.0e-3 ** 2)

    raw_move = abs(
        float(rescaled.norm().item()) - float(physical.norm().item())
    ) / float(physical.norm().item())
    weighted_first = flexibility_energy_norm(physical, stiffness)
    weighted_second = flexibility_energy_norm(rescaled, rescaled_stiffness)
    weighted_move = abs(weighted_second - weighted_first) / max(weighted_first, 1e-300)

    assert raw_move > 100.0, "the raw L2 norm should swing hugely under a unit change"
    assert weighted_move < 1e-12, "the flexibility norm must be unit-consistent"


def test_relative_gap_is_units_consistent_and_scale_invariant():
    dof = 6
    k = _spd(dof)
    a = torch.randn(dof, dtype=torch.float64)
    b = a + 0.1 * torch.randn(dof, dtype=torch.float64)
    gap = relative_flexibility_gap(a, b, k)
    assert gap > 0.0
    # Scaling both loads and the stiffness consistently leaves the gap unchanged.
    assert math.isclose(
        relative_flexibility_gap(4.0 * a, 4.0 * b, 4.0 * k), gap, rel_tol=1e-12
    )


def test_regularisation_rescues_a_singular_reference():
    """A free-free assembly (rigid-body nullspace) must not crash the metric."""

    k = torch.diag(torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64))
    r = torch.tensor([1.0, 0.0, 2.0], dtype=torch.float64)
    with pytest.raises((ValueError, RuntimeError)):
        flexibility_energy_norm(r, k)
    value = flexibility_energy_norm(r, k, regularisation=1.0e-6)
    assert math.isfinite(value) and value > 0.0


def test_shape_and_input_validation():
    k = _spd(4)
    with pytest.raises(ValueError):
        flexibility_energy_norm(torch.zeros(2, 2), k)
    with pytest.raises(ValueError):
        flexibility_energy_norm(torch.zeros(3), k)
    with pytest.raises(ValueError):
        flexibility_energy_norm(torch.zeros(4), k, regularisation=-1.0)
    # (1, dof) is accepted, matching how the solver carries generalized rows.
    assert flexibility_energy_norm(torch.zeros(1, 4, dtype=torch.float64), k) == 0.0
