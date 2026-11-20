"""B2 contract: TE-leg strength candidates are measured, never applied.

U2 from LOAD_REPRODUCTION_DIAGNOSIS: the trailing-edge leg of the last
chordwise row takes ``Gamma_n - Gamma_{n-1}`` (bound-rate history), while the
wake row inserted right after carries ``gamma_tev = Gamma_TE + gamma_lev``.
Those are different quantities, and the review explicitly forbids "fixing"
one into the other without an adjudicated shared-edge relation.

``te_leg_candidate_diagnostic`` (default OFF) assembles a SECOND packet for
the same state with the shed strength substituted on the TE legs and records
the resulting force delta.  This module pins:

* flag OFF -> no diagnostic payload, and the packet is the production one;
* flag ON  -> the payload exists, is finite, and the PRODUCTION packet
  (forces/moment/totals) is bit-identical to the flag-OFF packet;
* the candidate delta is a real number (nonzero when a LEV is being shed),
  so the next round has data to adjudicate with.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from test_q16_joint_deposited_flow_gpu import make_case

from fluxvortex.warp_fsi.q16_flux_v5m_native import NativeV5MConfig

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)

REQUIRED_KEYS = (
    "te_leg_baseline_strength_max_abs",
    "te_leg_shed_strength_max_abs",
    "te_leg_candidate_force_delta_norm_n",
    "te_leg_candidate_force_delta_relative",
    "te_leg_candidate_delta_n",
)


def _propose(alpha: float, diagnostic: bool, joint: bool = True):
    solver, stepper, owner, kin = make_case(alpha, joint)
    solver.settings = replace(
        solver.settings, te_leg_candidate_diagnostic=diagnostic
    )
    proposal = stepper.propose(
        owner,
        (kin.evaluate(solver.settings.aerodynamic_dt),),
        solver.settings.aerodynamic_dt,
    )
    return solver, proposal


def test_diagnostic_is_absent_by_default():
    _, proposal = _propose(19.0, diagnostic=False)
    diag = proposal.trial_state.diagnostics[-1]
    assert diag["te_leg_candidate"] == {}


def test_diagnostic_payload_is_finite_and_complete():
    _, proposal = _propose(19.0, diagnostic=True)
    diag = proposal.trial_state.diagnostics[-1]
    payload = diag["te_leg_candidate"]
    for key in REQUIRED_KEYS:
        assert key in payload, key
    for key in REQUIRED_KEYS[:-1]:
        assert torch.isfinite(torch.tensor(payload[key])).item(), key
    delta = payload["te_leg_candidate_delta_n"]
    assert len(delta) == 3
    assert all(torch.isfinite(torch.tensor(component)).item() for component in delta)
    assert payload["te_leg_candidate_force_delta_norm_n"] >= 0.0
    assert payload["te_leg_candidate_force_delta_relative"] >= 0.0


def test_diagnostic_does_not_perturb_the_production_packet():
    _, off = _propose(19.0, diagnostic=False)
    _, on = _propose(19.0, diagnostic=True)
    p_off = off.author_load.surface_load_packet
    p_on = on.author_load.surface_load_packet
    assert p_off is not None and p_on is not None
    assert torch.equal(p_off.point_forces_I, p_on.point_forces_I)
    assert torch.equal(p_off.total_force_I, p_on.total_force_I)
    assert torch.equal(p_off.total_moment_I, p_on.total_moment_I)
    assert torch.equal(off.load.total_force, on.load.total_force)


def test_candidate_delta_is_nonzero_while_lev_is_shed():
    """A separated strip sheds circulation, so the two candidates differ."""

    _, proposal = _propose(19.0, diagnostic=True)
    diag = proposal.trial_state.diagnostics[-1]
    payload = diag["te_leg_candidate"]
    assert diag["lev_release_count"] > 0, "A19 must release LEV for this test"
    assert payload["te_leg_shed_strength_max_abs"] > 0.0
    assert payload["te_leg_candidate_force_delta_norm_n"] > 0.0


@pytest.mark.parametrize("invalid", ["false", 0, 1, None])
def test_flag_requires_explicit_boolean(invalid):
    with pytest.raises(TypeError, match="te_leg_candidate_diagnostic"):
        NativeV5MConfig(
            chordwise_panels=2,
            spanwise_panels=2,
            density=1.0,
            freestream=1.0,
            aerodynamic_dt=1.0e-3,
            te_leg_candidate_diagnostic=invalid,
            device="cuda:0",
        )
