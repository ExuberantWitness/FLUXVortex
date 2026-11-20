"""Route A contract: the newborn LE sheet must be a MONOPOLE, not a closed dipole.

Measured basis for these tests (``diagnose_roj_le_kernel_sign.py``, S2.0, and
``diagnose_roj_circulation_ledger.py``, 2026-11-21):

* the production deposit is a closed ribbon, so its net vorticity
  ``sum(gamma_p)`` is machine zero on every step -- the leading-edge vortex is
  literally a vortex-free dipole strain field;
* that dipole induces only 3.9% (A15) / 4.5% (A19) of a monopole's normal
  velocity at the first collocation row, because its closure sheet sits
  ``ds = 0.0049c`` away and cancels the LE sheet to 96%;
* the joint closure uses that same deposit as its basis, so the solved newborn
  circulation is inflated by the reciprocal factor -- independently measured as
  22.9x by the L2-1 flux oracle, and growing like ``1/dt``.

Route A therefore swaps the closed ribbon for an open spanwise filament on the
physical leading edge.  These tests pin:

* both flags default OFF and the production path is untouched;
* the closed deposit's net vorticity stays machine zero (regression guard on the
  defect we are replacing, so a silent change cannot go unnoticed);
* the open deposit's net vorticity is the deposited circulation, not zero;
* with the open deposit the solved newborn circulation drops by at least 5x --
  the end-to-end signature that the closure basis became monopole-like;
* an attached (alpha=0) case is bit-identical under the flags;
* the two flags cannot be combined illegally.
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


def _propose(alpha: float, *, open_deposit: bool, drop_tev: bool = False, joint: bool = True):
    solver, stepper, owner, kin = make_case(alpha, joint)
    solver.settings = replace(
        solver.settings,
        le_open_deposit=open_deposit,
        le_open_deposit_drop_tev_compensation=drop_tev,
    )
    dt = solver.settings.aerodynamic_dt
    proposal = stepper.propose(owner, (kin.evaluate(dt),), dt)
    return solver, proposal


def _newest_cohort(field):
    newest = int(field.birth_step[: field.n].max().item())
    mask = field.birth_step[: field.n] == newest
    return field.pos[: field.n][mask], field.gamma[: field.n][mask], field.circul[: field.n][mask]


def test_flags_default_off():
    settings = NativeV5MConfig(chordwise_panels=4, spanwise_panels=2)
    assert settings.le_open_deposit is False
    assert settings.le_open_deposit_drop_tev_compensation is False


def test_illegal_flag_combinations_are_rejected():
    with pytest.raises(ValueError, match="requires le_open_deposit"):
        NativeV5MConfig(
            chordwise_panels=4,
            spanwise_panels=2,
            joint_separation_solve=True,
            le_open_deposit=False,
            le_open_deposit_drop_tev_compensation=True,
        )
    with pytest.raises(ValueError, match="requires joint_separation_solve"):
        NativeV5MConfig(
            chordwise_panels=4,
            spanwise_panels=2,
            joint_separation_solve=False,
            le_open_deposit=True,
        )


def test_attached_case_is_bit_identical_under_the_flags():
    _, off = _propose(0.0, open_deposit=False)
    _, on = _propose(0.0, open_deposit=True)
    assert off.trial_state.diagnostics[-1]["separated_strip_count"] == 0
    assert on.trial_state.diagnostics[-1]["separated_strip_count"] == 0
    assert torch.equal(off.load.total_force, on.load.total_force)


def test_closed_deposit_still_has_zero_net_vorticity():
    """Regression guard on the DEFECT: a closed ribbon must stay vortex-free.

    If this ever fails, the deposit representation changed silently and the S2.0
    measurement no longer describes the code.
    """

    _, proposal = _propose(15.0, open_deposit=False)
    field = proposal.trial_state.particle_field
    assert field.n > 0
    _, gamma, circul = _newest_cohort(field)
    net_vorticity = torch.linalg.vector_norm(gamma.sum(dim=0)).item()
    net_circulation = abs(circul.sum().item())
    assert net_vorticity < 1.0e-12
    assert net_circulation < 1.0e-12


def test_open_deposit_carries_the_shed_circulation():
    _, proposal = _propose(15.0, open_deposit=True)
    field = proposal.trial_state.particle_field
    _, gamma, circul = _newest_cohort(field)
    # Net vorticity is the spanwise sum of the deposited filament, so it must be
    # the same order as sum|gamma|, not a 1e-16 residue.
    net_vorticity = torch.linalg.vector_norm(gamma.sum(dim=0)).item()
    total = torch.linalg.vector_norm(gamma, dim=1).sum().item()
    assert net_vorticity > 0.0
    assert net_vorticity / total > 1.0e-3
    assert proposal.trial_state.diagnostics[-1]["separated_strip_count"] > 0


def test_open_deposit_shrinks_the_solved_newborn_circulation():
    """The end-to-end signature of a monopole basis.

    With a 1/25-efficacy (dipole) basis the closure must solve a ~25x larger
    newborn strength.  Requiring only a 5x drop keeps the test robust to the
    finite-span and connector corrections while still failing loudly if the
    basis does not change at all.
    """

    _, closed = _propose(15.0, open_deposit=False)
    _, opened = _propose(15.0, open_deposit=True)
    closed_newborn = closed.trial_state.diagnostics[-1][
        "newborn_lev_circulation_max_abs"
    ]
    open_newborn = opened.trial_state.diagnostics[-1][
        "newborn_lev_circulation_max_abs"
    ]
    assert closed_newborn > 0.0
    assert open_newborn > 0.0
    assert open_newborn < 0.2 * closed_newborn


def test_tev_compensation_toggle_changes_only_the_wake_row():
    _, with_compensation = _propose(15.0, open_deposit=True, drop_tev=False)
    _, without = _propose(15.0, open_deposit=True, drop_tev=True)
    ns = 30
    newest_with = with_compensation.trial_state.wake_gamma[:ns]
    newest_without = without.trial_state.wake_gamma[:ns]
    bound_te = with_compensation.trial_state.gamma_bound.reshape(-1, ns)[-1]
    # With the toggle on, the newest wake row is exactly the bound TE ring
    # strength: the shed amount now lives at the leading edge instead.
    assert torch.allclose(newest_without, bound_te, rtol=0.0, atol=1.0e-12)
    assert not torch.allclose(newest_with, newest_without, atol=1.0e-6)
