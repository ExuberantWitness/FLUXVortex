"""Regressions for the false physical acceptance in P1 stage 1.

Until the actual birth operator and material history are implemented, the
experimental path must reject inconsistent separated proposals. The flag is
explicitly enabled here; the pre-existing native suite uses its OFF default.
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
def test_separated_surrogate_cannot_pass_as_deposited_flow(alpha):
    solver, stepper, owner, kin = make_case(alpha, True)
    before = owner.state.digest()
    # The historical digest omits bank arrays; inspect them separately.
    bank_before = {k: v.clone() for k, v in vars(owner.state.source_bank).items()
                   if isinstance(v, torch.Tensor)}
    for _ in range(2):
        with pytest.raises(RuntimeError, match="joint deposited-flow Neumann rows failed"):
            stepper.propose(owner, (kin.evaluate(solver.settings.aerodynamic_dt),),
                            solver.settings.aerodynamic_dt)
        assert owner.state.digest() == before
        assert owner.state.step == 0
        assert owner.state.particle_field.n == 0
        for name, expected in bank_before.items():
            assert torch.equal(getattr(owner.state.source_bank, name), expected), name


def test_joint_attached_case_passes_and_matches_legacy():
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
    assert runs[0][0] == runs[1][0]
    assert torch.equal(runs[0][1], runs[1][1])


@pytest.mark.parametrize("invalid", ["false", 0, 1, None])
def test_joint_flag_requires_explicit_boolean(invalid):
    with pytest.raises(TypeError, match="joint_separation_solve"):
        replace(NativeV5MConfig(), joint_separation_solve=invalid)
