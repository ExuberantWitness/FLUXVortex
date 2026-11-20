"""E1-WINDOW contract: only the LE closure RHS is windowed, nothing else moves.

Rationale being tested (measured, not assumed): the pre-blow-up LE induction is
dominated by the ACCUMULATED wake (older 26%, rows 6-10 25%, rows 4-5 22%,
newest 3% at mean distance 1.3c), so far-field wake drives a LOCAL separation
judgement.  E1-WINDOW removes particles older than N steps from the LE closure
RHS only; the far wake stays fully active in the non-LE rows and in the loads.

These tests pin:
* window = 0 (production default) -> bit-identical to no window at all;
* window > 0 with old particles present -> the fraction is reported and the
  solved newborn differs (the model moved);
* attached case (no particles) -> window is a no-op.
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


def _warm(alpha: float, steps: int = 3, joint: bool = True):
    solver, stepper, owner, kin = make_case(alpha, joint)
    for step in range(1, steps + 1):
        proposal = stepper.propose(
            owner,
            (kin.evaluate(step * solver.settings.aerodynamic_dt),),
            solver.settings.aerodynamic_dt,
        )
        stepper.commit(owner, proposal)
    return solver, stepper, owner, kin


def _propose_from_state(solver, stepper, owner, kin, *, window: int):
    solver.settings = replace(solver.settings, le_constraint_window_steps=window)
    dt = solver.settings.aerodynamic_dt
    step = owner.state.step + 1
    return stepper.propose(owner, (kin.evaluate(step * dt),), dt)


def test_window_zero_is_the_unwindowed_model():
    solver, stepper, owner, kin = _warm(19.0)
    base = _propose_from_state(solver, stepper, owner, kin, window=0)
    diag = base.trial_state.diagnostics[-1]
    assert diag["le_constraint_window_steps"] == 0
    assert diag["le_window_old_particle_fraction"] == 0.0
    # and it must equal a proposal from an untouched (window=0) solver
    assert torch.equal(
        base.load.total_force,
        _propose_from_state(solver, stepper, owner, kin, window=0).load.total_force,
    )


def test_window_moves_only_the_le_closure():
    solver, stepper, owner, kin = _warm(19.0)
    plain = _propose_from_state(solver, stepper, owner, kin, window=0)
    windowed = _propose_from_state(solver, stepper, owner, kin, window=1)
    d_plain = plain.trial_state.diagnostics[-1]
    d_win = windowed.trial_state.diagnostics[-1]

    assert d_win["le_constraint_window_steps"] == 1
    assert d_win["le_window_old_particle_fraction"] > 0.5, (
        "with a 1-step window almost every particle is 'old'"
    )
    # the newborn strength must change (the LE closure RHS changed)
    assert not torch.equal(
        plain.trial_state.lev_strip_last_release,
        windowed.trial_state.lev_strip_last_release,
    ), "the windowed LE closure produced an identical shed strength"
    # the bound LE circulation stays pinned either way
    assert float(d_win["lesp_pin_max_abs"]) <= 1e-6


def test_window_is_a_noop_when_no_particles_exist():
    solver, stepper, owner, kin = _warm(0.0, steps=2)
    plain = _propose_from_state(solver, stepper, owner, kin, window=0)
    windowed = _propose_from_state(solver, stepper, owner, kin, window=100)
    d = windowed.trial_state.diagnostics[-1]
    assert d["separated_strip_count"] == 0
    assert d["le_window_old_particle_fraction"] == 0.0
    assert torch.equal(plain.load.total_force, windowed.load.total_force)


@pytest.mark.parametrize("invalid", ["100", -1, 1.5, None])
def test_window_requires_exact_nonnegative_int(invalid):
    with pytest.raises((TypeError, ValueError), match="le_constraint_window_steps"):
        NativeV5MConfig(
            chordwise_panels=2,
            spanwise_panels=2,
            density=1.0,
            freestream=1.0,
            aerodynamic_dt=1.0e-3,
            le_constraint_window_steps=invalid,
            device="cuda:0",
        )
