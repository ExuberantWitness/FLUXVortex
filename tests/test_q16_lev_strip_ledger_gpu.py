"""Leading-edge shedding ledger contract.

Why this exists (2026-11-20 post-mortem): the failed net-filament attempt took
``lambda_prev`` from a per-strip sum of the particle ``circul`` array.  That
array is an EDGE-level quantity repeated on every particle of the edge
(``particle_circulation = edge_circulation[edge_index]`` with no division by
the sub-division count), and the ribbon's three edge families are
``(+c, -c, connector differences)``; a per-strip sum is therefore a spanwise
DIFFERENCE of the sheet strength and the global sum is ~0.  Any quantity of
the form "lambda" must come from the deposit inputs, not from ``circul``.

``lev_strip_last_release`` is exactly the newest LE sheet row: the per-strip
strength (m^2/s, same measure as ``gamma_bound``) handed to the ribbon deposit
by the most recent step.  ``lev_strip_released_total`` is the running sum.

Both are history metadata: they must NOT enter ``NativeV5MState.digest()``,
so a ledger change alone cannot alter transaction/digest semantics.
"""

from __future__ import annotations

import pytest
import torch

from test_q16_joint_deposited_flow_gpu import make_case

from fluxvortex.warp_fsi.q16_flux_v5m_native import NativeV5MState

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


def _propose(alpha: float, steps: int = 1, joint: bool = True):
    solver, stepper, owner, kin = make_case(alpha, joint)
    last: list[torch.Tensor] = []
    for step in range(1, steps + 1):
        proposal = stepper.propose(
            owner,
            (kin.evaluate(step * solver.settings.aerodynamic_dt),),
            solver.settings.aerodynamic_dt,
        )
        state = proposal.trial_state
        assert state.lev_strip_last_release is not None
        last.append(state.lev_strip_last_release.clone())
        stepper.commit(owner, proposal)
    return solver, stepper, owner, last


def test_ledger_starts_zero_and_digest_ignores_it():
    solver, stepper, owner, kin = make_case(19.0, True)
    assert owner.state.lev_strip_last_release.abs().sum().item() == 0.0
    assert owner.state.lev_strip_released_total.abs().sum().item() == 0.0
    digest = owner.state.digest()
    mutated = owner.state.clone()
    mutated.lev_strip_last_release = torch.ones_like(
        mutated.lev_strip_last_release
    )
    mutated.lev_strip_released_total = torch.full_like(
        mutated.lev_strip_released_total, 123.0
    )
    assert mutated.digest() == digest, (
        "the shedding ledger is history metadata and must not enter the "
        "numerical digest"
    )


def test_attached_case_releases_nothing():
    _, _, owner, last = _propose(0.0, steps=2)
    assert all(float(v.abs().max().item()) == 0.0 for v in last)
    assert float(owner.state.lev_strip_released_total.abs().max().item()) == 0.0


def test_separated_case_records_the_newest_le_row():
    solver, _, owner, last = _propose(19.0, steps=1)
    diag = owner.state.diagnostics[-1]
    assert diag["lev_release_count"] > 0
    assert float(diag["lev_strip_last_release_max_abs"]) > 0.0
    # Same measure as the bound circulation: it must be a finite m^2/s-scale
    # number, not an edge-duplicated particle quantity.
    assert torch.isfinite(last[0]).all().item()
    assert float(last[0].abs().max().item()) < 10.0 * float(
        owner.state.gamma_bound.abs().max().item()
    )


def test_running_total_equals_the_sum_of_the_per_step_records():
    _, _, owner, last = _propose(19.0, steps=3)
    expected = torch.zeros_like(owner.state.lev_strip_released_total)
    for value in last:
        expected = expected + value
    assert torch.allclose(
        owner.state.lev_strip_released_total, expected, rtol=0.0, atol=1e-12
    )


def test_last_release_is_per_step_not_accumulated():
    """The per-step record must be the increment, not the running total."""

    _, _, owner, last = _propose(19.0, steps=3)
    assert float(last[0].abs().max().item()) > 0.0
    total = owner.state.lev_strip_released_total
    assert not torch.equal(last[-1], total), (
        "the per-step record must not equal the running total"
    )
    # And it must scale with a single step, not with the number of steps.
    assert float(last[-1].abs().max().item()) < float(total.abs().max().item())
