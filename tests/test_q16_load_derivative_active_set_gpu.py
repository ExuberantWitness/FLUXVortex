"""B1 contract: load derivative operators use the ACTIVE constraint set.

Background (LOAD_REPRODUCTION_DIAGNOSIS §3 / P1_SHARED_EDGE review Q-B): the
bound circulation is solved on a constrained system (``separated_aic`` for the
legacy pinned path, the pinned block for the joint path), but the load
derivatives -- ``Mf1 = rho * P(q) A^-1 N(q)`` and the substep
``velocity_force`` (Mf2_1) -- were linearized with the raw all-Neumann AIC.
On a separated state those are different systems, so the derivatives did not
describe the equation that produced gamma.

``load_derivative_active_set`` (default OFF) redirects the assembler to the
active set.  This module pins what a rigid-wing propose CAN observe:

* the reported scope for each path (unconstrained / separated-pinned-exact /
  pinned-ignoring-newborn-coupling);
* that the flag leaves the INSTANTANEOUS load untouched -- it only changes
  derivative operators (Mf1/Mf2_1), which the rigid path never evaluates
  inside propose(); a membrane FSI run is required to see the difference;
* the exact-bool type contract.
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


def _propose(alpha: float, joint: bool, active_set: bool):
    solver, stepper, owner, kin = make_case(alpha, joint)
    solver.settings = replace(
        solver.settings, load_derivative_active_set=active_set
    )
    proposal = stepper.propose(
        owner,
        (kin.evaluate(solver.settings.aerodynamic_dt),),
        solver.settings.aerodynamic_dt,
    )
    return solver, proposal


def test_scope_is_unconstrained_by_default():
    _, proposal = _propose(19.0, joint=False, active_set=False)
    diag = proposal.trial_state.diagnostics[-1]
    assert diag["load_derivative_scope"] == "unconstrained"


def test_scope_reports_separated_pinned_when_enabled():
    _, proposal = _propose(19.0, joint=False, active_set=True)
    diag = proposal.trial_state.diagnostics[-1]
    assert diag["separated_strip_count"] > 0, "A19 must separate for this test"
    assert diag["load_derivative_scope"] == "separated_pinned_exact"


def test_scope_reports_joint_pinned_coupling_omitted():
    _, proposal = _propose(19.0, joint=True, active_set=True)
    diag = proposal.trial_state.diagnostics[-1]
    assert diag["load_derivative_scope"] == "pinned_ignores_newborn_coupling"


def test_attached_state_stays_unconstrained_even_when_enabled():
    _, proposal = _propose(0.0, joint=False, active_set=True)
    diag = proposal.trial_state.diagnostics[-1]
    assert diag["separated_strip_count"] == 0
    assert diag["load_derivative_scope"] == "unconstrained"


def test_flag_does_not_touch_the_instantaneous_load():
    """Only derivative operators move; the resolved packet must be identical.

    The unified point-load packet is built from geometry, gamma and
    mf2_history -- never from aic -- so switching the derivative operator may
    not leak into the load the solver reports for this proposal.
    """

    _, off = _propose(19.0, joint=False, active_set=False)
    _, on = _propose(19.0, joint=False, active_set=True)
    p_off = off.author_load.surface_load_packet
    p_on = on.author_load.surface_load_packet
    assert p_off is not None and p_on is not None
    assert torch.equal(p_off.point_forces_I, p_on.point_forces_I)
    assert torch.equal(p_off.total_force_I, p_on.total_force_I)
    assert torch.equal(p_off.total_moment_I, p_on.total_moment_I)


@pytest.mark.parametrize("invalid", ["false", 0, 1, None])
def test_flag_requires_explicit_boolean(invalid):
    with pytest.raises(TypeError, match="load_derivative_active_set"):
        NativeV5MConfig(
            chordwise_panels=2,
            spanwise_panels=2,
            density=1.0,
            freestream=1.0,
            aerodynamic_dt=1.0e-3,
            load_derivative_active_set=invalid,
            device="cuda:0",
        )
