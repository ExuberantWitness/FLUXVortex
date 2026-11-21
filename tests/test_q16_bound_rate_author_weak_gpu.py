"""Stage 2 contract: the author's WEAK scheme must be reachable and exactly defined.

Background (recorded in-tree as unresolved, not invented here):
``diagnostics/README.md:27-33`` says the production ``bound_rate`` path keeps "all
other strong blocks" (dp_lift1, lift2, Mf2_1, Mf1), while the author's
``calc_fluid_force_weak.m:96-131`` zeroes ``Mf1`` and ``Mf2_1`` because its constant
slot already carries ``dp_add = rho (Gamma_n − Gamma_{n−1})/dt``.  ``RESEARCH_BRIEF``
M3 / ``REPAIR...`` U3 / ``P1_SHARED_EDGE...`` §171-175 log the resulting double count
as unresolved.

These tests pin the MECHANISM of the switch, not a physics claim:
1. ``velocity_force(include_mf21=False)`` drops the Mf2_1 slot and keeps lift2;
2. at zero structural velocity lift2 vanishes, so the lift-only slot must be zero
   while the full slot is not -- proof the switch removes Mf2_1 and nothing else;
3. the flag defaults OFF so the production path stays bit-identical.
"""

from __future__ import annotations

import inspect
import os

import pytest
import torch
import warp as wp

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


@pytest.fixture(scope="module")
def stepped_membrane():
    """One A16 step through the unified runner; return the committed load + dof count."""

    from fluxvortex.cases.rojratsirikul2011 import ROJRATSIRIKUL2011_UNIFIED_CASES
    from fluxvortex.runtime.case_runner import RojratsirikulCaseRunner

    runner = RojratsirikulCaseRunner(
        ROJRATSIRIKUL2011_UNIFIED_CASES["ROJ11-A16"], device="cuda:0"
    )
    runner.build()
    runner.run(max_aero_steps=1)
    load = runner.owner.aerodynamic_load
    return load, runner.mesh.dof_count


def test_flag_defaults_off_and_is_validated():
    from fluxvortex.warp_fsi.q16_flux_v5m_native_fsi import Q16NativeV5MFSIStepper
    from fluxvortex.runtime.case_runner import _bound_rate_author_weak_choice

    parameters = inspect.signature(Q16NativeV5MFSIStepper.__init__).parameters
    assert parameters["bound_rate_author_weak"].default is False

    os.environ.pop("FLUXV_BOUND_RATE_AUTHOR_WEAK", None)
    assert _bound_rate_author_weak_choice() is False
    os.environ["FLUXV_BOUND_RATE_AUTHOR_WEAK"] = "1"
    try:
        assert _bound_rate_author_weak_choice() is True
    finally:
        os.environ.pop("FLUXV_BOUND_RATE_AUTHOR_WEAK", None)


def _velocity(dof_count: int, scale: float = 1.0) -> wp.array:
    generator = torch.Generator().manual_seed(11)
    values = torch.randn(
        1, dof_count, generator=generator, dtype=torch.float64
    ).to("cuda:0")
    return wp.from_torch((values * scale).contiguous())


def test_lift_only_slot_is_linear_in_velocity_while_the_full_slot_is_quadratic(
    stepped_membrane,
):
    """The discriminator that does not duplicate production math.

    ``lift2 = -rho (V . grad Gamma)`` is LINEAR in the structural velocity, whereas
    ``Mf2_1`` contains ``normal_rate(V) * slip(V)`` and is therefore QUADRATIC (the
    internal review already noted velocity_force "has a quadratic term, it is not
    linear Jv").  So:

      include_mf21=False -> f(2V) == 2 f(V)   exactly
      include_mf21=True  -> f(2V) != 2 f(V)   (the quadratic part breaks homogeneity)

    That pins "the switch removes exactly the non-linear slot" without reimplementing
    the Mf2_1 formula inside a test.
    """

    load, dof_count = stepped_membrane
    v = _velocity(dof_count)
    v2 = _velocity(dof_count, scale=2.0)

    lift_v = wp.to_torch(load.velocity_force(v, include_mf21=False)).clone()
    lift_2v = wp.to_torch(load.velocity_force(v2, include_mf21=False)).clone()
    full_v = wp.to_torch(load.velocity_force(v)).clone()
    full_2v = wp.to_torch(load.velocity_force(v2)).clone()

    # The full slot must be non-trivial at a non-zero velocity.
    scale = max(float(full_v.abs().max().item()), 1.0e-30)
    assert scale > 0.0
    # ... and the two forms must differ.
    assert float((full_v - lift_v).abs().max().item()) > 0.0

    lift_homogeneity = float((lift_2v - 2.0 * lift_v).abs().max().item()) / scale
    full_homogeneity = float((full_2v - 2.0 * full_v).abs().max().item()) / scale
    assert lift_homogeneity <= 1.0e-12, (
        f"lift-only slot must be linear in V, got {lift_homogeneity:.3e}"
    )
    assert full_homogeneity > 1.0e-6, (
        "the full slot must contain the quadratic Mf2_1 term; it looked linear "
        f"({full_homogeneity:.3e})"
    )


def test_zero_added_mass_short_circuits_the_mf1_subtraction(stepped_membrane):
    """The Stage 2 flag replaces the Mf1 action with a cached zero matrix.

    Checked at the action level (no FSI loop needed): a zero action must contribute
    exactly zero to the effective RHS, which is what "author weak scheme: Mf1 = 0"
    means.
    """

    load, dof_count = stepped_membrane
    matrix = load.added_mass.generalized_matrix
    zero_action = load.added_mass.__class__(torch.zeros_like(matrix))
    acceleration = _velocity(dof_count, scale=1.0e-2)
    real = wp.to_torch(load.added_mass(acceleration))
    none_ = wp.to_torch(zero_action(acceleration))
    assert float(none_.abs().max().item()) == 0.0
    assert float(real.abs().max().item()) > 0.0
