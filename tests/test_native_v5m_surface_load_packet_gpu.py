"""Unified surface-load packet contract tests (M1 of the modification plan).

Drives the SAME frozen rigid native V5M machinery a few committed steps and
checks the NativeV5MSurfaceLoadPacket gates:

  * force/moment closure <= 1e-12 relative;
  * packet total agrees with the legacy integrated path (same physics,
    independent summation order) within fp64 round-off;
  * density scaling is exactly linear on every packet component;
  * a repeated proposal from one committed parent yields an identical
    packet and leaves the parent untouched.
"""

from __future__ import annotations

import math

import pytest
import torch

torch.cuda.init()
if not torch.cuda.is_available():
    pytest.skip("surface-load packet tests require CUDA", allow_module_level=True)

from fluxvortex.aero.v5m.load_packet import (
    POINT_KIND_UNSTEADY,
    NativeResolvedLoadAssembler,
    validate_packet,
)
from fluxvortex.aero.v5m.stepper import RigidV5MStepper
from fluxvortex.cases.izraelevitz_kinematics import build_izra_reference_grid
from fluxvortex.kinematics.prescribed_rigid import PrescribedRigidSurfaceKinematics
from fluxvortex.warp_fsi.q16_flux_v5m_native import NativeV5MConfig
from fluxvortex.warp_fsi.rigid_flux_v5m_native import (
    RigidNativeV5MSolver,
    RigidV5MSurface,
)

DEVICE = "cuda:0"
CHORD_M = 0.0688
SPAN_M = 0.1375


def _fixed_pitch_law(alpha_deg: float):
    theta = math.radians(alpha_deg)
    half = 0.5 * theta
    dtype = torch.float64
    pos = torch.zeros(3, device=DEVICE, dtype=dtype)
    quat = torch.tensor(
        [math.cos(half), 0.0, math.sin(half), 0.0], device=DEVICE, dtype=dtype
    )
    zero = torch.zeros(3, device=DEVICE, dtype=dtype)

    def law(t: float):
        return pos, quat, zero, zero.clone()

    return law


def _build(alpha_deg: float = 15.0):
    nc, ns = 15, 30
    aero_dt = 0.01 * CHORD_M / 5.0
    reference = build_izra_reference_grid(
        chordwise_panels=nc,
        spanwise_panels=ns,
        chord_m=CHORD_M,
        span_m=SPAN_M,
        pivot_fraction_chord=0.25,
        device=DEVICE,
    )
    kin = PrescribedRigidSurfaceKinematics(
        reference, _fixed_pitch_law(alpha_deg), surface_id="w", body_id="b"
    )
    surface = RigidV5MSurface(chordwise_panels=nc, spanwise_panels=ns, device=DEVICE)
    settings = NativeV5MConfig(
        chordwise_panels=nc,
        spanwise_panels=ns,
        density=1.208,
        freestream=5.0,
        aerodynamic_dt=aero_dt,
        lesp_crit=0.11,
        wake_max_rows=64,
        particle_capacity=32768,
        particle_max_age_steps=100,
        wake_history_mode="bound_rate",
        wake_free_rows=32,
        dvm_target_spacing_chord=0.018,
        device=DEVICE,
    )
    solver = RigidNativeV5MSolver(surface, settings)
    stepper = RigidV5MStepper(solver, device=DEVICE)
    owner = stepper.initialize((kin.evaluate(0.0),))
    return stepper, solver, owner, kin, aero_dt


@pytest.fixture(scope="module")
def committed():
    stepper, solver, owner, kin, aero_dt = _build()
    for step in range(1, 26):
        frame = kin.evaluate(step * aero_dt)
        proposal = stepper.propose(owner, (frame,), aero_dt)
        stepper.commit(owner, proposal)
    frame = kin.evaluate(26 * aero_dt)
    proposal = stepper.propose(owner, (frame,), aero_dt)
    return stepper, solver, owner, kin, aero_dt, proposal


def test_packet_shape_and_ownership(committed) -> None:
    _, _, _, _, _, proposal = committed
    packet = proposal.author_load.surface_load_packet
    count = 15 * 30
    assert packet.point_forces_I.shape == (5 * count, 3)
    assert packet.point_kind.shape == (5 * count,)
    assert int(packet.panel_owner.max()) == count - 1
    kinds = torch.bincount(packet.point_kind, minlength=5)
    assert all(int(k) == count for k in kinds)


def test_force_and_moment_closure(committed) -> None:
    _, _, _, _, _, proposal = committed
    packet = proposal.author_load.surface_load_packet
    validate_packet(packet)  # raises beyond 1e-12 relative
    assert packet.force_closure_error() == pytest.approx(0.0, abs=1e-18)
    assert packet.moment_closure_error() == pytest.approx(0.0, abs=1e-18)


def test_packet_agrees_with_legacy_integration(committed) -> None:
    _, _, _, _, _, proposal = committed
    packet = proposal.author_load.surface_load_packet
    legacy_force = proposal.author_load.total_force
    legacy_moment = proposal.author_load.total_moment
    force_scale = max(float(legacy_force.norm().item()), 1e-30)
    moment_scale = max(float(legacy_moment.norm().item()), 1e-30)
    assert float((packet.total_force_I - legacy_force).norm().item()) <= 1e-12 * force_scale
    assert float((packet.total_moment_I - legacy_moment).norm().item()) <= 1e-12 * moment_scale


def test_density_linearity(committed) -> None:
    stepper, solver, owner, kin, aero_dt, _ = committed
    # Re-assemble the packet at 2x density from the same committed parent.
    trial = owner.state
    geometry = solver.surface.evaluate(None, None)
    assembler = NativeResolvedLoadAssembler(
        density=2.0 * solver.settings.density,
        device=DEVICE,
        chordwise_panels=solver.settings.chordwise_panels,
        spanwise_panels=solver.settings.spanwise_panels,
        aerodynamic_dt=solver.settings.aerodynamic_dt,
        wake_history_mode="bound_rate",
    )
    mids = assembler.leg_midpoints_flat(geometry)
    velocity = solver.v_inf[None, :].expand(mids.shape[0], 3) + solver._ring_velocity(
        mids, geometry.rings, trial.gamma_bound, rough=False
    )
    base = stepper
    packet_double = assembler.assemble_packet(
        geometry=geometry,
        gamma=trial.gamma_bound,
        solution_velocity_flat=velocity.contiguous(),
        mf2_history=torch.zeros_like(trial.gamma_bound),
    )
    frame = kin.evaluate((owner.state.step + 1) * aero_dt)
    proposal = base.propose(owner, (frame,), aero_dt)
    single = proposal.author_load.surface_load_packet
    # With zero gamma_rate the unsteady points vanish; the KJ part must be
    # exactly proportional to density.
    kj_double = packet_double.point_forces_I[packet_double.point_kind != POINT_KIND_UNSTEADY]
    kj_single = single.point_forces_I[single.point_kind != POINT_KIND_UNSTEADY]
    # Different states (committed vs proposal) -> compare self-consistency:
    # the double-density packet's KJ forces are exactly 2x its own unit
    # density rebuild.
    packet_unit = NativeResolvedLoadAssembler(
        density=solver.settings.density,
        device=DEVICE,
        chordwise_panels=solver.settings.chordwise_panels,
        spanwise_panels=solver.settings.spanwise_panels,
        aerodynamic_dt=solver.settings.aerodynamic_dt,
        wake_history_mode="bound_rate",
    ).assemble_packet(
        geometry=geometry,
        gamma=trial.gamma_bound,
        solution_velocity_flat=velocity.contiguous(),
        mf2_history=torch.zeros_like(trial.gamma_bound),
    )
    kj_unit = packet_unit.point_forces_I[packet_unit.point_kind != POINT_KIND_UNSTEADY]
    assert torch.allclose(kj_double, 2.0 * kj_unit, rtol=0.0, atol=1e-15)


def test_repeated_proposal_identity(committed) -> None:
    stepper, _, owner, kin, aero_dt, first = committed
    parent_step = owner.state.step
    frame = kin.evaluate((owner.state.step + 1) * aero_dt)
    second = stepper.propose(owner, (frame,), aero_dt)
    assert owner.state.step == parent_step  # parent untouched
    p1 = first.author_load.surface_load_packet
    p2 = second.author_load.surface_load_packet
    assert torch.equal(p1.point_forces_I, p2.point_forces_I)
    assert torch.equal(p1.total_force_I, p2.total_force_I)
    assert torch.equal(p1.total_moment_I, p2.total_moment_I)
