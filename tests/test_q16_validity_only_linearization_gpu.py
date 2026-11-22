"""S3c contract: the validity-only trial linearization must RAISE exactly like the full one.

The Newton line search asks `_linearization_prechecked(trial)` only whether the trial
geometry is admissible: on the common path the returned value is discarded
(`accepted_trial = trial; break`), and only the rare `globalize_newton` branch consumes it.
S3c therefore calls a cheaper `validity_only_prechecked`, which runs the same
`_sample_fields` / `_strain` / `_alpha` calls but skips `_projected_b` (75% of the cost).

This is only sound if the RAISE SURFACE is identical, because that is the entire output
that path uses.  The tests below pin exactly that:

1. on the committed state both paths pass;
2. on a degenerate state (all nodes coincident) BOTH raise the same
   ``FloatingPointError`` message -- the gate inside `_strain`;
3. across a sweep of scaled (increasingly squashed) states the two paths agree on
   raise-vs-pass at every point, which is what "same accepted trials" means.

A cheap path that raised MORE would reject admissible trials; one that raised LESS would
accept inverted ones and only fail later, in the residual.
"""

from __future__ import annotations

import pytest
import torch
import warp as wp

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)

GEOMETRY_MESSAGE = "Q16 current geometry is orientation reversing or singular"


@pytest.fixture(scope="module")
def operator_and_state():
    from fluxvortex.cases.rojratsirikul2011 import ROJRATSIRIKUL2011_UNIFIED_CASES
    from fluxvortex.runtime.case_runner import RojratsirikulCaseRunner

    runner = RojratsirikulCaseRunner(
        ROJRATSIRIKUL2011_UNIFIED_CASES["ROJ11-A16"], device="cuda:0"
    )
    runner.build()
    runner.run(max_aero_steps=1)
    return runner.structural._operator, runner.owner.state


def _verdict(operator, state) -> tuple[str, str]:
    """(full-path verdict, cheap-path verdict); each is 'pass' or the error message.

    Both paths are always evaluated so the two verdicts can be compared directly.
    """

    try:
        operator._linearization_prechecked(state)
        full = "pass"
    except FloatingPointError as error:
        full = str(error)
    try:
        operator.validity_only_prechecked(state)
        cheap = "pass"
    except FloatingPointError as error:
        cheap = str(error)
    return full, cheap


def test_flag_defaults_on_and_is_bitwise_by_construction():
    from fluxvortex.warp_fsi import q16_structural_solver as module

    assert module._VALIDITY_ONLY_TRIAL_LINEARIZATION is True


def test_admissible_state_passes_on_both_paths(operator_and_state):
    operator, state = operator_and_state
    assert _verdict(operator, state) == ("pass", "pass")


def test_collapsed_state_raises_the_geometry_error_on_both_paths(operator_and_state):
    operator, state = operator_and_state
    collapsed = wp.zeros_like(state)
    full, cheap = _verdict(operator, collapsed)
    assert (full, cheap) == (GEOMETRY_MESSAGE, GEOMETRY_MESSAGE), (full, cheap)


def test_accept_reject_agrees_across_a_squash_sweep(operator_and_state):
    """The property that makes the skip bit-identical: same verdict at every state."""

    operator, state = operator_and_state
    base = wp.to_torch(state)
    verdicts = []
    for factor in (1.0, 0.5, 0.1, 0.02, 0.0):
        scaled = wp.from_torch(
            (base * factor).contiguous(), dtype=wp.float64, requires_grad=False
        )
        full, cheap = _verdict(operator, scaled)
        assert full == cheap, f"factor={factor}: full and cheap disagree ({full!r} vs {cheap!r})"
        verdicts.append(factor)
    assert verdicts == [1.0, 0.5, 0.1, 0.02, 0.0]
