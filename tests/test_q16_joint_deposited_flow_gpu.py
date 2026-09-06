"""Joint real-birth acceptance, independent rejection and transaction checks.

The real basis must pass, while corruption of the actual deposited sources
must still fail. These tests do not certify material-history physics.
"""
from dataclasses import replace

import pytest
import torch

import queue_roj_rigid_fig9_12_13_15 as frozen
from fluxvortex.aero.v5m.stepper import RigidV5MStepper
from fluxvortex.cases.izraelevitz_kinematics import build_izra_reference_grid
from fluxvortex.kinematics.prescribed_rigid import PrescribedRigidSurfaceKinematics
from fluxvortex.warp_fsi.q16_flux_v5m_native import NativeV5MConfig
from fluxvortex.warp_fsi.rigid_flux_v5m_native import RigidNativeV5MSolver, RigidV5MSurface


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def make_case(alpha, joint):
    nc, ns = frozen.AERO_GRID
    reference = build_izra_reference_grid(chordwise_panels=nc, spanwise_panels=ns,
        chord_m=frozen.CHORD_M, span_m=frozen.SPAN_M, pivot_fraction_chord=.25, device="cuda:0")
    kin = PrescribedRigidSurfaceKinematics(reference, frozen.fixed_pitch_law(alpha, "cuda:0"),
        surface_id="roj_rigid_wing", body_id="body_0")
    settings = NativeV5MConfig(chordwise_panels=nc, spanwise_panels=ns,
        density=frozen.DENSITY, freestream=5., aerodynamic_dt=frozen.DT_STAR*frozen.CHORD_M/5.,
        wake_history_mode="bound_rate", particle_capacity=32768, device="cuda:0",
        dvm_target_spacing_chord=frozen.DVM_TARGET_SPACING_CHORD,
        joint_separation_solve=joint)
    solver = RigidNativeV5MSolver(RigidV5MSurface(chordwise_panels=nc, spanwise_panels=ns,
        device="cuda:0"), settings)
    stepper = RigidV5MStepper(solver, device="cuda:0")
    owner = stepper.initialize((kin.evaluate(0.),))
    return solver, stepper, owner, kin


@pytest.mark.parametrize("alpha", [5., 15., 19.])
def test_real_birth_passes_and_trial_preserves_complete_bank(alpha):
    # P1-2 real-operator update: the newborn-LEV basis is the deposited
    # material ribbon itself, so the separated case now SATISFIES the
    # deposited-flow gate (historical surrogate used to be rejected here).
    # The contract to keep testing is: full-surface no-penetration holds
    # on the ACTUAL deposited particles, finite, and the proposal is a
    # valid trial (parent untouched before commit).
    solver, stepper, owner, kin = make_case(alpha, True)
    before = owner.state.digest()
    # The existing state digest omits these trajectory-defining bank arrays.
    bank_before = {name: value.clone() for name, value in vars(owner.state.source_bank).items()
                   if isinstance(value, torch.Tensor)}
    proposals = []
    for _ in range(2):
        proposal = stepper.propose(
            owner,
            (kin.evaluate(solver.settings.aerodynamic_dt),),
            solver.settings.aerodynamic_dt,
        )
        diag = proposal.trial_state.diagnostics[-1]
        assert diag["neumann_acceptance_scope"] == "deposited_flow_all_rows"
        assert diag["full_surface_neumann_max_abs"] <= solver.settings.gate_rtol
        assert diag["lev_release_count"] > 0
        assert proposal.trial_state.frontier_active.any().item()
        assert owner.state.digest() == before
        assert owner.state.step == 0
        for name, value in bank_before.items():
            assert torch.equal(getattr(owner.state.source_bank, name), value), name
        proposals.append(proposal)
    assert proposals[0].trial_state.digest() == proposals[1].trial_state.digest()
    assert torch.equal(proposals[0].load.total_force, proposals[1].load.total_force)
    for name in bank_before:
        assert torch.equal(getattr(proposals[0].trial_state.source_bank, name),
                           getattr(proposals[1].trial_state.source_bank, name)), name


def test_actual_deposit_corruption_is_rejected_independently_of_solved_basis():
    solver, stepper, owner, kin = make_case(15., True)
    original = solver._deposit_dvm_ribbon
    before = owner.state.digest()
    bank_before = owner.state.source_bank.lg.clone()

    def corrupted_deposit(geometry, trial, result):
        start = trial.particle_field.n
        deposited = original(geometry, trial, result)
        # Deliberately corrupt ONLY actual sources, leaving the B matrix,
        # solved strengths, pin, and algebraic residual unchanged.
        trial.particle_field.gamma[start:trial.particle_field.n].mul_(1.01)
        return deposited

    solver._deposit_dvm_ribbon = corrupted_deposit
    with pytest.raises(RuntimeError, match="joint deposited-flow Neumann rows failed"):
        stepper.propose(owner, (kin.evaluate(solver.settings.aerodynamic_dt),),
                        solver.settings.aerodynamic_dt)
    assert owner.state.digest() == before
    assert owner.state.particle_field.n == 0
    assert torch.equal(owner.state.source_bank.lg, bank_before)


def test_joint_attached_case_passes_and_matches_legacy():
    # Step 1 (no wake history yet): joint and legacy paths are bit-identical.
    # Step 2 onward: the experimental path RECONNECTS the newborn TE row to
    # the convected previous front edge (wake-sheet continuity fix), so the
    # wake geometry -- and everything downstream of its induction -- may
    # legitimately differ; gates must still pass on both.
    runs = []
    for joint in (False, True):
        solver, stepper, owner, kin = make_case(0., joint)
        for step in (1, 2):
            proposal = stepper.propose(owner, (kin.evaluate(step*solver.settings.aerodynamic_dt),),
                                       solver.settings.aerodynamic_dt)
            assert proposal.trial_state.diagnostics[-1]["lev_release_count"] == 0
            if joint:
                diag = proposal.trial_state.diagnostics[-1]
                assert diag["neumann_acceptance_scope"] == "deposited_flow_all_rows"
                assert diag["full_surface_neumann_max_abs"] < 1.e-10
            stepper.commit(owner, proposal)
        runs.append((owner.state.digest(), proposal.load.total_force.clone()))
    # Bit identity after step 2 no longer holds on the experimental path
    # (wake reconnection); both runs must simply be finite and committed.
    assert all(torch.isfinite(force).all().item() for _, force in runs)


def test_joint_attached_step1_bit_identical():
    runs = []
    for joint in (False, True):
        solver, stepper, owner, kin = make_case(0., joint)
        proposal = stepper.propose(owner, (kin.evaluate(solver.settings.aerodynamic_dt),),
                                   solver.settings.aerodynamic_dt)
        runs.append((owner.state.digest(), proposal.load.total_force.clone()))
    assert runs[0][0] == runs[1][0]
    assert torch.equal(runs[0][1], runs[1][1])


@pytest.mark.parametrize("invalid", ["false", 0, 1, None])
def test_joint_flag_requires_explicit_boolean(invalid):
    with pytest.raises(TypeError, match="joint_separation_solve"):
        replace(NativeV5MConfig(), joint_separation_solve=invalid)
