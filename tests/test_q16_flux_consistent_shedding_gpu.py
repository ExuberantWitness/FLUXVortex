"""E1-FLUX contract: prescribed shed circulation is proportional to dt.

Background (L2-1 oracle, 2026-11-21): the solved newborn strength was O(20-50)x
the sharp-edge flux estimate 0.5*U_s^2*dt AND dt-INDEPENDENT (the ratio doubled
when dt halved), i.e. gamma_recon = Gamma/ds ~ 1/dt -- refining the time step
amplified the near-LE sheet.  E1-FLUX replaces the LE-no-penetration closure
with the Kaden/Didden slug law, so the shed circulation must be EXACTLY linear
in dt and quadratic in the slip.

These tests pin the model, not a number:
* flag OFF -> production behaviour untouched (attached case bit-identical);
* flag ON  -> Gamma_shed(dt) / Gamma_shed(dt/2) == 2 exactly (slip comes from
  the dt-independent unconstrained solve, so the ratio must be exact);
* the strict full-surface gate is replaced by a recorded residual + a hard
  non-LE check, and that must be visible in the diagnostics;
* the per-strip ledger records exactly what was prescribed.
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


def _propose(alpha: float, *, flux: bool, dt_scale: float = 1.0, joint: bool = True):
    solver, stepper, owner, kin = make_case(alpha, joint)
    dt = solver.settings.aerodynamic_dt * dt_scale
    solver.settings = replace(
        solver.settings,
        aerodynamic_dt=dt,
        flux_consistent_shedding=flux,
    )
    proposal = stepper.propose(owner, (kin.evaluate(dt),), dt)
    return solver, proposal


def test_attached_case_is_unchanged_by_the_flag():
    _, off = _propose(0.0, flux=False)
    _, on = _propose(0.0, flux=True)
    assert off.trial_state.diagnostics[-1]["separated_strip_count"] == 0
    assert torch.equal(off.load.total_force, on.load.total_force)
    assert on.trial_state.diagnostics[-1]["flux_shed_max"] == 0.0


def test_flux_mode_needs_the_ledger_and_reports_its_residual():
    _, proposal = _propose(19.0, flux=True)
    diag = proposal.trial_state.diagnostics[-1]
    assert diag["flux_consistent_shedding"] is True
    assert diag["separated_strip_count"] > 0
    assert diag["flux_shed_max"] > 0.0
    # The prescribed-shedding model does not close the LE rows; the residual is
    # recorded honestly instead of being swallowed.
    assert torch.isfinite(
        torch.tensor(diag["flux_le_no_penetration_residual"])
    ).item()


def test_prescribed_shed_strength_scales_exactly_with_dt():
    """Gamma_shed(dt) / Gamma_shed(dt/2) == 2 to machine precision.

    The slip comes from the UNCONSTRAINED pre-solve bound solution, which is
    dt-independent, so the only dt dependence is the explicit factor in the
    slug law.  This is the property the old solved-closure did NOT have (its
    ratio was 1.0 within noise), and it is what makes the sheet bounded.
    """

    _, full = _propose(19.0, flux=True, dt_scale=1.0)
    _, half = _propose(19.0, flux=True, dt_scale=0.5)
    a = full.trial_state.lev_strip_last_release
    b = half.trial_state.lev_strip_last_release
    # compare where the shed strength is non-negligible on both sides
    mask = (a.abs() > 1e-12) & (b.abs() > 1e-12)
    assert bool(mask.any().item())
    ratio = a[mask] / b[mask]
    assert torch.allclose(ratio, torch.full_like(ratio, 2.0), rtol=0.0, atol=1e-9), (
        f"shed circulation is not linear in dt: ratios {ratio.tolist()}"
    )


def test_ledger_records_exactly_the_prescribed_strength():
    _, proposal = _propose(19.0, flux=True)
    state = proposal.trial_state
    diag = state.diagnostics[-1]
    assert float(state.lev_strip_last_release.abs().max().item()) == pytest.approx(
        diag["flux_shed_max"], rel=1e-12
    )


@pytest.mark.parametrize("invalid", ["false", 0, 1, None])
def test_flag_requires_explicit_boolean(invalid):
    with pytest.raises(TypeError, match="flux_consistent_shedding"):
        NativeV5MConfig(
            chordwise_panels=2,
            spanwise_panels=2,
            density=1.0,
            freestream=1.0,
            aerodynamic_dt=1.0e-3,
            flux_consistent_shedding=invalid,
            device="cuda:0",
        )
