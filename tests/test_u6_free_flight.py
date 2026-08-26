"""U6: GeneralizedLoadPacket, J^T f projection, full velocity chain, free flight.

Plan §5.8/§7.3/§10/§14 U6: the SAME surface forces project via the
kinematic Jacobian transpose into body/joint/elastic generalized loads
with a work-conjugacy audit (surface power = generalized power); the
body+joint+Q16 kinematic chain gains its full velocity content (body
translation, ω×r, joint rate, elastic rate); and
``PartitionedFreeFlightFSI`` composes aero + composite dynamics under
one global transaction with exactly one formal commit per subsystem per
step.
"""
from __future__ import annotations

import dataclasses
import math
from types import SimpleNamespace

import pytest
import torch

from fluxvortex.coupling.free_flight import PartitionedFreeFlightFSI
from fluxvortex.dynamics.composite import CompositeBodyJointQ16Dynamics
from fluxvortex.dynamics.joints import JointState, PrescribedJointDynamics
from fluxvortex.dynamics.load_packet import (
    GeneralizedLoadPacket,
    SurfaceLoad,
    build_generalized_load_packet,
    project_surface_loads_to_body,
)
from fluxvortex.dynamics.rigid_body_se3 import (
    RigidBodySE3Dynamics,
    RigidBodyState,
)
from fluxvortex.kinematics.body_joint_q16 import BodyJointQ16Kinematics
from fluxvortex.kinematics.frames import SurfaceFrame
from fluxvortex.state.world import WorldDynamicState, WorldOwner

DTYPE = torch.float64

DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda:0")


def _vec(values, device):
    return torch.tensor(values, device=device, dtype=DTYPE)


# ---------------------------------------------------------------------------
# SurfaceLoad / project_surface_loads_to_body
# ---------------------------------------------------------------------------

class TestSurfaceLoad:
    @pytest.mark.parametrize("device", DEVICES)
    def test_from_panels_reduces_resultant_about_origin(self, device):
        points = _vec([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device)
        forces = _vec([[0.0, 2.0, 0.0], [3.0, 0.0, 0.0]], device)
        load = SurfaceLoad.from_panels("wing_0", forces, points)
        assert load.surface_id == "wing_0"
        torch.testing.assert_close(load.total_force_I, _vec([3.0, 2.0, 0.0], device))
        # M_origin = sum r x f = [0,0,2] + [0,0,-3] = [0,0,-1]
        torch.testing.assert_close(load.total_moment_I, _vec([0.0, 0.0, -1.0], device))
        load.validate()

    @pytest.mark.parametrize("device", DEVICES)
    def test_from_panels_rejects_shape_mismatch(self, device):
        forces = _vec([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device)
        points = _vec([[0.0, 0.0, 0.0]], device)
        with pytest.raises(ValueError, match="panel_points_I"):
            SurfaceLoad.from_panels("wing_0", forces, points)


class TestProjectSurfaceLoadsToBody:
    @pytest.mark.parametrize("device", DEVICES)
    def test_single_surface_moment_shift_to_com(self, device):
        points = _vec([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device)
        forces = _vec([[0.0, 2.0, 0.0], [3.0, 0.0, 0.0]], device)
        load = SurfaceLoad.from_panels("wing_0", forces, points)

        wrench = project_surface_loads_to_body((load,), _vec([0.0, 0.0, 0.0], device))
        # About the origin COM: force unchanged, moment unchanged.
        torch.testing.assert_close(wrench["body_0"]["force_I"], _vec([3.0, 2.0, 0.0], device))
        torch.testing.assert_close(wrench["body_0"]["moment_I"], _vec([0.0, 0.0, -1.0], device))

        # Body COM at [0.5, 0, 0]: M_body = M_origin - r_body x F
        # r x F = [0.5,0,0]x[3,2,0] = [0,0,1]  =>  M_body = [0,0,-2]
        wrench = project_surface_loads_to_body((load,), _vec([0.5, 0.0, 0.0], device))
        torch.testing.assert_close(wrench["body_0"]["force_I"], _vec([3.0, 2.0, 0.0], device))
        torch.testing.assert_close(wrench["body_0"]["moment_I"], _vec([0.0, 0.0, -2.0], device))

    @pytest.mark.parametrize("device", DEVICES)
    def test_two_surfaces_sum_then_shift(self, device):
        load_a = SurfaceLoad.from_panels(
            "wing_0",
            _vec([[0.0, 2.0, 0.0]], device), _vec([[1.0, 0.0, 0.0]], device))
        load_b = SurfaceLoad.from_panels(
            "wing_1",
            _vec([[3.0, 0.0, 0.0]], device), _vec([[0.0, 1.0, 0.0]], device))
        r_body = _vec([0.5, 0.0, 0.0], device)
        wrench = project_surface_loads_to_body((load_a, load_b), r_body)
        torch.testing.assert_close(wrench["body_0"]["force_I"], _vec([3.0, 2.0, 0.0], device))
        # Identical to the single-surface two-panel case: projection is
        # independent of how panels are grouped into surfaces.
        torch.testing.assert_close(wrench["body_0"]["moment_I"], _vec([0.0, 0.0, -2.0], device))

    @pytest.mark.parametrize("device", DEVICES)
    def test_force_through_com_gives_zero_moment(self, device):
        # Force applied exactly at the body COM: moment about COM vanishes.
        load = SurfaceLoad.from_panels(
            "wing_0", _vec([[2.0, -1.0, 4.0]], device), _vec([[1.0, 2.0, 3.0]], device))
        wrench = project_surface_loads_to_body((load,), _vec([1.0, 2.0, 3.0], device))
        torch.testing.assert_close(
            wrench["body_0"]["moment_I"], _vec([0.0, 0.0, 0.0], device), rtol=0.0, atol=1e-15)

    @pytest.mark.parametrize("device", DEVICES)
    def test_custom_body_id_key(self, device):
        load = SurfaceLoad.from_panels(
            "wing_0", _vec([[1.0, 0.0, 0.0]], device), _vec([[0.0, 0.0, 0.0]], device))
        wrench = project_surface_loads_to_body((load,), _vec([0.0, 0.0, 0.0], device), body_id="fuselage")
        assert set(wrench) == {"fuselage"}


# ---------------------------------------------------------------------------
# Work conjugacy (plan §5.8 acceptance)
# ---------------------------------------------------------------------------

class TestWorkConjugacy:
    @pytest.mark.parametrize("device", DEVICES)
    def test_rigid_surface_power_equals_body_power(self, device):
        # Rigidly-moving surface: v_p = v_com + omega_I x (r_p - r_com).
        # Identity: sum f.(omega x rho) = omega . sum(rho x f) = omega.M_com.
        torch.manual_seed(0)
        r_com = _vec([0.3, -0.2, 0.1], device)
        v_com = _vec([0.7, 0.1, -0.4], device)
        omega_B = _vec([0.2, -0.5, 0.9], device)
        quat = _vec([math.cos(0.26), 0.0, 0.0, math.sin(0.26)], device)
        body = dataclasses.replace(
            RigidBodyState.identity("body_0", 2.0, (1.0, 2.0, 3.0), device=device),
            position_I=r_com, quaternion_IB=quat,
            linear_velocity_I=v_com, angular_velocity_B=omega_B)
        R_IB = RigidBodySE3Dynamics().rotation_matrix(quat)
        omega_I = R_IB @ omega_B

        points = _vec([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.4, -0.6, 0.9]], device)
        forces = _vec([[0.0, 2.0, 0.0], [3.0, 0.0, 0.0], [-1.0, 0.5, 2.0]], device)
        velocities = v_com + torch.cross(
            omega_I.expand_as(points), points - r_com, dim=-1)
        load = SurfaceLoad.from_panels("wing_0", forces, points, velocities)

        state = WorldDynamicState(elastic_states={}, body_states={"body_0": body}, joint_states={})
        packet = build_generalized_load_packet((load,), state)

        expected_surface = float((forces * velocities).sum().item())
        wrench = packet.body_wrenches_I["body_0"]
        expected_general = (
            float((wrench["force_I"] * v_com).sum().item())
            + float((wrench["moment_I"] * omega_I).sum().item()))
        assert packet.surface_power == pytest.approx(expected_surface, rel=1e-13)
        assert packet.generalized_power == pytest.approx(expected_general, rel=1e-13)
        assert packet.relative_work_error < 1e-12
        assert packet.audit_passes(tolerance=1e-10)

    @pytest.mark.parametrize("device", DEVICES)
    def test_elastic_power_matches_exact_jt_projection(self, device):
        # 1-DOF elastic mode: point x = q * d_p, local velocity qd * d_p.
        # The exact J^T f is Q = sum f.d_p, and Q.qd = surface power.
        qd = 0.6
        dirs = _vec([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device)
        forces = _vec([[0.5, 2.0, -1.0], [3.0, 0.0, 0.4]], device)
        q = 0.25
        points = q * dirs
        velocities = qd * dirs
        load = SurfaceLoad.from_panels("wing_0", forces, points, velocities)

        body = RigidBodyState.identity("body_0", 1.0, (1.0, 1.0, 1.0), device=device)
        elastic = SimpleNamespace(q=torch.tensor([q], device=device, dtype=DTYPE),
                                  qd=torch.tensor([qd], device=device, dtype=DTYPE))
        state = WorldDynamicState(
            elastic_states={"wing_0": elastic},
            body_states={"body_0": body}, joint_states={})
        Q_e = float((forces * dirs).sum(dim=-1).sum().item())
        packet = build_generalized_load_packet(
            (load,), state, elastic_forces={"wing_0": torch.tensor([Q_e], device=device, dtype=DTYPE)})

        expected_surface = qd * Q_e
        assert packet.surface_power == pytest.approx(expected_surface, rel=1e-13)
        assert packet.generalized_power == pytest.approx(expected_surface, rel=1e-13)
        assert packet.relative_work_error < 1e-12

    @pytest.mark.parametrize("device", DEVICES)
    def test_full_chain_body_plus_elastic_power(self, device):
        # v_p = v_com + omega x rho + u_local: the surface power splits
        # exactly into body wrench power + elastic power.
        v_com = _vec([0.3, -0.2, 0.1], device)
        omega = _vec([0.1, 0.2, -0.3], device)
        body = dataclasses.replace(
            RigidBodyState.identity("body_0", 1.5, (1.0, 1.0, 1.0), device=device),
            position_I=_vec([0.0, 0.0, 0.0], device),
            linear_velocity_I=v_com, angular_velocity_B=omega)

        rho = _vec([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device)
        forces = _vec([[0.0, 2.0, 0.0], [3.0, 0.0, 0.0]], device)
        u_local = _vec([[0.05, -0.02, 0.01], [-0.03, 0.04, 0.0]], device)
        qd = 0.8
        velocities = v_com + torch.cross(omega.expand_as(rho), rho, dim=-1) + u_local
        load = SurfaceLoad.from_panels("wing_0", forces, rho, velocities)

        # Exact J^T f for the elastic coordinate (u_local = qd * shape):
        shape = u_local / qd
        Q_e = float((forces * shape).sum(dim=-1).sum().item())
        elastic = SimpleNamespace(q=torch.zeros(1, device=device, dtype=DTYPE),
                                  qd=torch.tensor([qd], device=device, dtype=DTYPE))
        state = WorldDynamicState(
            elastic_states={"wing_0": elastic},
            body_states={"body_0": body}, joint_states={})
        packet = build_generalized_load_packet(
            (load,), state, elastic_forces={"wing_0": torch.tensor([Q_e], device=device, dtype=DTYPE)})

        expected_surface = float((forces * velocities).sum().item())
        assert packet.surface_power == pytest.approx(expected_surface, rel=1e-13)
        assert packet.generalized_power == pytest.approx(expected_surface, rel=1e-13)
        assert packet.relative_work_error < 1e-12

    @pytest.mark.parametrize("device", DEVICES)
    def test_missing_panel_velocities_fail_closed(self, device):
        load = SurfaceLoad(
            surface_id="wing_0",
            panel_forces_I=_vec([[1.0, 0.0, 0.0]], device),
            total_force_I=_vec([1.0, 0.0, 0.0], device),
            total_moment_I=_vec([0.0, 0.0, 0.0], device),
        )
        body = RigidBodyState.identity("body_0", 1.0, (1.0, 1.0, 1.0), device=device)
        state = WorldDynamicState(elastic_states={}, body_states={"body_0": body}, joint_states={})
        with pytest.raises(ValueError, match="work-conjugacy audit"):
            build_generalized_load_packet((load,), state)
        # Explicitly disabling the audit degrades gracefully to zero power.
        packet = build_generalized_load_packet((load,), state, audit_velocities=False)
        assert packet.surface_power == 0.0
        assert packet.generalized_power == 0.0
        assert packet.relative_work_error == 0.0

    @pytest.mark.parametrize("device", DEVICES)
    def test_empty_loads_give_zero_packet(self, device):
        body = RigidBodyState.identity("body_0", 1.0, (1.0, 1.0, 1.0), device=device)
        state = WorldDynamicState(elastic_states={}, body_states={"body_0": body}, joint_states={})
        packet = build_generalized_load_packet((), state)
        torch.testing.assert_close(
            packet.body_wrenches_I["body_0"]["force_I"],
            torch.zeros(3, device=device, dtype=DTYPE))
        assert packet.surface_power == 0.0
        assert packet.generalized_power == 0.0
        assert packet.relative_work_error == 0.0

    @pytest.mark.parametrize("device", DEVICES)
    def test_zero_load_factory(self, device):
        packet = GeneralizedLoadPacket.zero_load(device=device)
        torch.testing.assert_close(
            packet.body_wrenches_I["body_0"]["force_I"],
            torch.zeros(3, device=device, dtype=DTYPE))
        torch.testing.assert_close(
            packet.body_wrenches_I["body_0"]["moment_I"],
            torch.zeros(3, device=device, dtype=DTYPE))
        assert packet.audit_passes()


# ---------------------------------------------------------------------------
# CompositeBodyJointQ16Dynamics.propose under the packet contract
# ---------------------------------------------------------------------------

class _RecordingElasticAdapter:
    def __init__(self) -> None:
        self.calls = []

    def propose(self, committed, loads, dt):
        self.calls.append((loads, dt))
        return "elastic_proposed"


class TestCompositeProposeWithPacket:
    @pytest.mark.parametrize("device", DEVICES)
    def test_zero_load_packet_leaves_body_unchanged(self, device):
        body_dyn = RigidBodySE3Dynamics()

        def law(t):
            return _vec([0.1 * t], device), _vec([0.1], device)

        joint = PrescribedJointDynamics("right_flap", "body_0", law)
        elastic = _RecordingElasticAdapter()
        composite = CompositeBodyJointQ16Dynamics(body_dyn, [joint], {"wing_0": elastic})

        body_state = RigidBodyState.identity("body_0", 2.0, (1.0, 2.0, 3.0), device=device)
        committed = WorldDynamicState(
            elastic_states={"wing_0": object()},
            body_states={"body_0": body_state},
            joint_states={})
        advanced = composite.propose(
            committed, GeneralizedLoadPacket.zero_load(device=device), dt=1e-3, time=0.5)

        new_body = advanced["body_states"]["body_0"]
        torch.testing.assert_close(new_body.position_I, body_state.position_I, rtol=0.0, atol=0.0)
        torch.testing.assert_close(new_body.quaternion_IB, body_state.quaternion_IB, rtol=0.0, atol=0.0)
        torch.testing.assert_close(new_body.linear_velocity_I, body_state.linear_velocity_I, rtol=0.0, atol=0.0)
        torch.testing.assert_close(new_body.angular_velocity_B, body_state.angular_velocity_B, rtol=0.0, atol=0.0)
        # Prescribed joint evaluated at t + dt.
        new_joint = advanced["joint_states"]["right_flap"]
        torch.testing.assert_close(new_joint.coordinates, _vec([0.1 * 0.501], device))
        # Elastic adapter driven once with its (zero) load.
        assert len(elastic.calls) == 1
        assert elastic.calls[0][0] is None
        assert elastic.calls[0][1] == 1e-3

    @pytest.mark.parametrize("device", DEVICES)
    def test_packet_wrench_rotates_moment_into_body_frame(self, device):
        # Body rotated 90 deg about z: R_IB maps body x -> inertial y, so an
        # inertial moment [1,0,0] becomes body-frame moment [0,-1,0].
        theta = math.pi / 2
        quat = _vec([math.cos(theta / 2), 0.0, 0.0, math.sin(theta / 2)], device)
        mass, I_y = 2.0, 2.0
        body_state = dataclasses.replace(
            RigidBodyState.identity("body_0", mass, (1.0, I_y, 3.0), device=device),
            quaternion_IB=quat)
        committed = WorldDynamicState(
            elastic_states={}, body_states={"body_0": body_state}, joint_states={})

        force_I = _vec([0.0, 3.0, 0.0], device)
        moment_I = _vec([1.0, 0.0, 0.0], device)
        packet = GeneralizedLoadPacket(
            body_wrenches_I={"body_0": {"force_I": force_I, "moment_I": moment_I}})

        composite = CompositeBodyJointQ16Dynamics(RigidBodySE3Dynamics(), [], {})
        dt = 1e-3
        advanced = composite.propose(committed, packet, dt=dt, time=0.0)
        new_body = advanced["body_states"]["body_0"]

        # Semi-implicit increments from rest:
        #   dv_I = F/m dt (inertial),  domega_B = I^-1 M_B dt.
        torch.testing.assert_close(
            new_body.linear_velocity_I, force_I * (dt / mass), rtol=0.0, atol=1e-15)
        torch.testing.assert_close(
            new_body.angular_velocity_B, _vec([0.0, -dt / I_y, 0.0], device),
            rtol=0.0, atol=1e-15)
        torch.testing.assert_close(
            new_body.position_I, new_body.linear_velocity_I * dt, rtol=0.0, atol=1e-15)

    @pytest.mark.parametrize("device", DEVICES)
    def test_elastic_forces_routed_by_id(self, device):
        body_state = RigidBodyState.identity("body_0", 1.0, (1.0, 1.0, 1.0), device=device)
        committed = WorldDynamicState(
            elastic_states={"wing_0": object(), "wing_1": object()},
            body_states={"body_0": body_state}, joint_states={})
        wing0_force = _vec([1.0, 2.0], device)
        wing1_force = _vec([3.0], device)
        packet = GeneralizedLoadPacket(
            elastic_forces={"wing_0": wing0_force, "wing_1": wing1_force})

        wing0, wing1 = _RecordingElasticAdapter(), _RecordingElasticAdapter()
        composite = CompositeBodyJointQ16Dynamics(
            RigidBodySE3Dynamics(), [], {"wing_0": wing0, "wing_1": wing1})
        advanced = composite.propose(committed, packet, dt=2e-3, time=0.0)

        assert wing0.calls == [(wing0_force, 2e-3)]
        assert wing1.calls == [(wing1_force, 2e-3)]
        assert advanced["elastic_states"] == {
            "wing_0": "elastic_proposed", "wing_1": "elastic_proposed"}

    @pytest.mark.parametrize("device", DEVICES)
    def test_legacy_none_loads_still_advance(self, device):
        # U5 compatibility: loads=None means zero load everywhere.
        body_state = RigidBodyState.identity("body_0", 2.0, (1.0, 1.0, 1.0), device=device)
        committed = WorldDynamicState(
            elastic_states={"wing_0": object()},
            body_states={"body_0": body_state}, joint_states={})
        elastic = _RecordingElasticAdapter()
        composite = CompositeBodyJointQ16Dynamics(RigidBodySE3Dynamics(), [], {"wing_0": elastic})
        advanced = composite.propose(committed, None, dt=1e-3, time=0.0)
        new_body = advanced["body_states"]["body_0"]
        torch.testing.assert_close(new_body.position_I, body_state.position_I)
        torch.testing.assert_close(new_body.linear_velocity_I, body_state.linear_velocity_I)
        assert len(elastic.calls) == 1


# ---------------------------------------------------------------------------
# BodyJointQ16Kinematics: full position + velocity chain (plan §7.2)
# ---------------------------------------------------------------------------

class _MockQ16LocalAdapter:
    """One flat square panel + LE/TE lines in the surface reference frame."""

    def __init__(self, device, with_velocities=True):
        rings = _vec([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], device)
        leading = _vec([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device)
        trailing = _vec([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]], device)
        if with_velocities:
            ring_vel = 0.1 * rings
            colloc_vel = _vec([[0.1, 0.05, -0.2]], device)
            lead_vel = 0.05 * leading
            trail_vel = 0.05 * trailing
        else:
            ring_vel = torch.zeros_like(rings)
            colloc_vel = torch.zeros(1, 3, device=device, dtype=DTYPE)
            lead_vel = torch.zeros_like(leading)
            trail_vel = torch.zeros_like(trailing)
        self.frame = SurfaceFrame(
            surface_id="wing_0", body_id="body_0",
            panel_rings_I=rings.reshape(1, 4, 3),
            panel_ring_velocity_I=ring_vel.reshape(1, 4, 3),
            collocation_I=_vec([[0.5, 0.5, 0.0]], device),
            collocation_velocity_I=colloc_vel,
            normals_I=_vec([[0.0, 0.0, 1.0]], device),
            areas=torch.tensor([1.0], device=device, dtype=DTYPE),
            leading_edge_I=leading,
            trailing_edge_I=trailing,
            leading_velocity_I=lead_vel,
            trailing_velocity_I=trail_vel,
            chordwise_panels=1,
            spanwise_panels=1,
            topology_digest="mock-q16:1",
        )

    def evaluate(self, q, qd):
        return self.frame


def _identity_joint(device):
    def joint(joint_state):
        return torch.zeros(3, device=device, dtype=DTYPE), torch.eye(3, device=device, dtype=DTYPE)
    return joint


def _offset_rotated_joint(device):
    # Origin offset [0.5, 0, 0], rotation R_x(90 deg): (x,y,z)->(x,-z,y).
    def joint(joint_state):
        return (_vec([0.5, 0.0, 0.0], device),
                _vec([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], device))
    return joint


def _zero_joint_state(device):
    return JointState("j0", "body_0",
                      torch.zeros(1, device=device, dtype=DTYPE),
                      torch.zeros(1, device=device, dtype=DTYPE))


def _make_kinematics(device, joint=None, joint_velocity=None, with_velocities=True):
    return BodyJointQ16Kinematics(
        _MockQ16LocalAdapter(device, with_velocities=with_velocities),
        joint if joint is not None else _identity_joint(device),
        surface_id="wing_0", body_id="body_0",
        joint_velocity=joint_velocity)


def _elastic_state(device):
    return SimpleNamespace(q=torch.zeros(2, device=device, dtype=DTYPE),
                           qd=torch.zeros(2, device=device, dtype=DTYPE))


class TestBodyJointQ16KinematicsChain:
    @pytest.mark.parametrize("device", DEVICES)
    def test_identity_body_and_joint_match_local_frame(self, device):
        kin = _make_kinematics(device)
        body = RigidBodyState.identity("body_0", 1.0, (1.0, 1.0, 1.0), device=device)
        local = kin._q16.frame
        frame = kin.evaluate(body, _zero_joint_state(device), _elastic_state(device), time=0.0)
        assert frame.surface_id == "wing_0"
        assert frame.body_id == "body_0"
        # Exact arithmetic: identity transforms are bit-for-bit.
        for name in ("panel_rings_I", "panel_ring_velocity_I", "collocation_I",
                     "collocation_velocity_I", "normals_I", "areas",
                     "leading_edge_I", "trailing_edge_I",
                     "leading_velocity_I", "trailing_velocity_I"):
            assert torch.equal(getattr(frame, name), getattr(local, name)), name
        assert frame.chordwise_panels == local.chordwise_panels
        assert frame.spanwise_panels == local.spanwise_panels
        assert frame.topology_digest == local.topology_digest

    @pytest.mark.parametrize("device", DEVICES)
    def test_translated_body_shifts_points_and_adds_translation_velocity(self, device):
        kin = _make_kinematics(device)
        offset = _vec([1.0, 2.0, 3.0], device)
        v_com = _vec([0.5, -0.25, 0.75], device)
        body = dataclasses.replace(
            RigidBodyState.identity("body_0", 1.0, (1.0, 1.0, 1.0), device=device),
            position_I=offset, linear_velocity_I=v_com)
        local = kin._q16.frame
        frame = kin.evaluate(body, _zero_joint_state(device), _elastic_state(device), time=0.1)

        torch.testing.assert_close(
            frame.collocation_I, local.collocation_I + offset, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            frame.panel_rings_I, local.panel_rings_I + offset, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            frame.leading_edge_I, local.leading_edge_I + offset, rtol=0.0, atol=0.0)
        # omega = 0: velocity = v_com + local velocity, exactly.
        torch.testing.assert_close(
            frame.collocation_velocity_I, local.collocation_velocity_I + v_com,
            rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            frame.panel_ring_velocity_I, local.panel_ring_velocity_I + v_com,
            rtol=0.0, atol=0.0)

    @pytest.mark.parametrize("device", DEVICES)
    def test_rotating_body_velocity_contains_omega_cross_r(self, device):
        # 90 deg z-rotation, omega_B = [0,0,2], no translation, no local
        # velocity, no joint motion: v_I = omega_I x r_I exactly.
        kin = _make_kinematics(device, with_velocities=False)
        theta = math.pi / 2
        quat = _vec([math.cos(theta / 2), 0.0, 0.0, math.sin(theta / 2)], device)
        omega_B = _vec([0.0, 0.0, 2.0], device)
        body = dataclasses.replace(
            RigidBodyState.identity("body_0", 1.0, (1.0, 1.0, 1.0), device=device),
            quaternion_IB=quat, angular_velocity_B=omega_B)
        frame = kin.evaluate(body, _zero_joint_state(device), _elastic_state(device), time=0.0)

        R = RigidBodySE3Dynamics().rotation_matrix(quat)
        local = kin._q16.frame
        # Position chain: x_I = R_IB x_S (body at origin, identity joint).
        torch.testing.assert_close(
            frame.collocation_I, (R @ local.collocation_I.T).T, rtol=0.0, atol=1e-15)
        # collocation [0.5, 0.5, 0] -> R maps to [-0.5, 0.5, 0].
        torch.testing.assert_close(
            frame.collocation_I[0], _vec([-0.5, 0.5, 0.0], device), rtol=0.0, atol=1e-15)
        # omega_I = R [0,0,2] = [0,0,2]; v = omega x r = [-1, -1, 0].
        torch.testing.assert_close(
            frame.collocation_velocity_I[0], _vec([-1.0, -1.0, 0.0], device),
            rtol=0.0, atol=1e-15)
        # Same identity for every ring node: v = omega_I x (x_I - x_com).
        expected = torch.cross(
            (R @ omega_B).expand_as(frame.panel_rings_I.reshape(-1, 3)),
            frame.panel_rings_I.reshape(-1, 3), dim=-1)
        torch.testing.assert_close(
            frame.panel_ring_velocity_I.reshape(-1, 3), expected, rtol=0.0, atol=1e-15)
        # Normals rotate with the body.
        torch.testing.assert_close(
            frame.normals_I, local.normals_I @ R.T, rtol=0.0, atol=1e-15)

    @pytest.mark.parametrize("device", DEVICES)
    def test_joint_offset_and_rotation_compose_positions(self, device):
        kin = _make_kinematics(device, joint=_offset_rotated_joint(device), with_velocities=False)
        body = RigidBodyState.identity("body_0", 1.0, (1.0, 1.0, 1.0), device=device)
        frame = kin.evaluate(body, _zero_joint_state(device), _elastic_state(device), time=0.0)
        # collocation [0.5,0.5,0] --R_x90--> [0.5,0,0.5] --+p_BJ--> [1.0,0,0.5].
        torch.testing.assert_close(
            frame.collocation_I[0], _vec([1.0, 0.0, 0.5], device), rtol=0.0, atol=1e-15)
        # Normal [0,0,1] rotates to [0,-1,0] under R_x(90 deg).
        torch.testing.assert_close(
            frame.normals_I[0], _vec([0.0, -1.0, 0.0], device), rtol=0.0, atol=1e-15)

    @pytest.mark.parametrize("device", DEVICES)
    def test_joint_rate_velocity_term(self, device):
        # pdot_BJ = [0.1,0,0], omega_JB = [0,0,0.3]: the velocity must
        # include pdot + omega_JB x (R_BJ x_S).
        def joint_velocity(joint_state):
            return _vec([0.1, 0.0, 0.0], device), _vec([0.0, 0.0, 0.3], device)

        kin = _make_kinematics(
            device, joint=_offset_rotated_joint(device),
            joint_velocity=joint_velocity, with_velocities=False)
        body = RigidBodyState.identity("body_0", 1.0, (1.0, 1.0, 1.0), device=device)
        rates = _vec([0.4], device)
        joint_state = JointState("j0", "body_0", _vec([0.2], device), rates)
        frame = kin.evaluate(body, joint_state, _elastic_state(device), time=0.0)
        # collocation: R_BJ x_S = [0.5,0,0.5];
        # omega_J x (R_BJ x_S) = [0,0,0.3]x[0.5,0,0.5] = [0, 0.15, 0].
        torch.testing.assert_close(
            frame.collocation_velocity_I[0], _vec([0.1, 0.15, 0.0], device),
            rtol=0.0, atol=1e-15)

    @pytest.mark.parametrize("device", DEVICES)
    def test_full_chain_velocity_formula(self, device):
        # Independent recomputation of
        # v = v_com + R_IB omega_B x (x_I - x_com) + R_IB (pdot + omega_JB x (R_BJ x_S) + R_BJ u_S)
        theta = 0.4
        quat = _vec([math.cos(theta / 2), 0.0, 0.0, math.sin(theta / 2)], device)
        v_com = _vec([0.2, -0.1, 0.05], device)
        omega_B = _vec([0.3, 0.2, -0.4], device)
        body = dataclasses.replace(
            RigidBodyState.identity("body_0", 1.0, (1.0, 1.0, 1.0), device=device),
            position_I=_vec([0.7, -0.3, 0.2], device),
            quaternion_IB=quat,
            linear_velocity_I=v_com,
            angular_velocity_B=omega_B)

        def joint_velocity(joint_state):
            return _vec([0.05, 0.0, -0.02], device), _vec([0.0, 0.1, 0.3], device)

        kin = _make_kinematics(
            device, joint=_offset_rotated_joint(device), joint_velocity=joint_velocity)
        frame = kin.evaluate(body, _zero_joint_state(device), _elastic_state(device), time=0.0)

        R_IB = RigidBodySE3Dynamics().rotation_matrix(quat)
        p_BJ, R_BJ = _offset_rotated_joint(device)(_zero_joint_state(device))
        pdot, omega_J = joint_velocity(None)
        x_S = kin._q16.frame.collocation_I[0]
        u_S = kin._q16.frame.collocation_velocity_I[0]
        x_B = p_BJ + R_BJ @ x_S
        x_I = body.position_I + R_IB @ x_B
        expected_pos = body.position_I + R_IB @ (p_BJ + R_BJ @ x_S)
        expected_vel = (v_com
                        + torch.cross(R_IB @ omega_B, x_I - body.position_I, dim=0)
                        + R_IB @ (pdot + torch.cross(omega_J, R_BJ @ x_S, dim=0) + R_BJ @ u_S))
        torch.testing.assert_close(frame.collocation_I[0], expected_pos, rtol=0.0, atol=1e-14)
        torch.testing.assert_close(frame.collocation_velocity_I[0], expected_vel, rtol=0.0, atol=1e-14)

    @pytest.mark.parametrize("device", DEVICES)
    def test_nonzero_joint_rates_without_mapping_fail_closed(self, device):
        kin = _make_kinematics(device)  # no joint_velocity installed
        body = RigidBodyState.identity("body_0", 1.0, (1.0, 1.0, 1.0), device=device)
        joint_state = JointState("j0", "body_0", _vec([0.2], device), _vec([0.5], device))
        with pytest.raises(ValueError, match="joint_velocity"):
            kin.evaluate(body, joint_state, _elastic_state(device), time=0.0)

    @pytest.mark.parametrize("device", DEVICES)
    def test_surface_ids_property_composes(self, device):
        # The chain keeps the adapter's surface identity for aggregation.
        kin = _make_kinematics(device)
        body = RigidBodyState.identity("body_0", 1.0, (1.0, 1.0, 1.0), device=device)
        frame = kin.evaluate(body, _zero_joint_state(device), _elastic_state(device), time=0.0)
        assert frame.surface_id == "wing_0"
        assert frame.body_id == "body_0"


# ---------------------------------------------------------------------------
# PartitionedFreeFlightFSI
# ---------------------------------------------------------------------------

class _MockFreeFlightKinematics:
    """Duck-typed multi-surface provider: returns the guessed body state."""

    def __init__(self):
        self.calls = []

    def evaluate(self, dynamic_state, time):
        self.calls.append(time)
        return dynamic_state.body_states["body_0"]


class _MockSpringAero:
    """Single-panel aero mock: force k (L - x) e_x at a fixed body arm.

    Panel velocities are consistent with the guessed body twist, so the
    work-conjugacy gate sees an exact rigid projection.
    """

    def __init__(self, stiffness, target, device):
        self.k = stiffness
        self.L = target
        self.device = device
        self.proposals = []
        self.commits = []

    def propose(self, aero_state, frames, dt):
        body = frames  # the guessed RigidBodyState
        x = float(body.position_I[0].item())
        force = _vec([self.k * (self.L - x), 0.0, 0.0], self.device)
        arm = _vec([1.0, 0.0, 0.0], self.device)
        point = body.position_I + arm
        R_IB = RigidBodySE3Dynamics().rotation_matrix(body.quaternion_IB)
        omega_I = R_IB @ body.angular_velocity_B
        velocity = body.linear_velocity_I + torch.cross(omega_I, arm, dim=0)
        load = SurfaceLoad.from_panels(
            "wing_0", force.reshape(1, 3), point.reshape(1, 3), velocity.reshape(1, 3))
        proposal = SimpleNamespace(surface_loads=(load,))
        self.proposals.append(proposal)
        return proposal

    def commit(self, owner, proposal):
        self.commits.append(proposal)


class _MockEmptyAero:
    def __init__(self):
        self.proposals = []
        self.commits = []

    def propose(self, aero_state, frames, dt):
        proposal = SimpleNamespace(surface_loads=())
        self.proposals.append(proposal)
        return proposal

    def commit(self, owner, proposal):
        self.commits.append(proposal)


class _ExpandingDynamics:
    """Non-convergent mock: multiplies the guessed position each trial."""

    def __init__(self):
        self.calls = 0

    def propose(self, committed, loads, dt, time):
        self.calls += 1
        body = committed.body_states["body_0"]
        return {
            "body_states": {
                "body_0": dataclasses.replace(body, position_I=4.0 * body.position_I)},
            "joint_states": {},
            "elastic_states": {},
        }


def _make_owner(body, device):
    return WorldOwner(
        dynamic_state=WorldDynamicState(
            elastic_states={}, body_states={"body_0": body}, joint_states={}),
        aero_state=object(),
        previous_load=None,
        generation=0,
    )


class TestPartitionedFreeFlightFSI:
    @pytest.mark.parametrize("device", DEVICES)
    def test_spring_coupled_step_converges_and_commits_once(self, device):
        mass = 1.0
        stiffness = 2.0e3
        target = 1.0
        x0 = 0.2
        dt = 1e-3
        body = dataclasses.replace(
            RigidBodyState.identity("body_0", mass, (1.0, 1.0, 1.0), device=device),
            position_I=_vec([x0, 0.0, 0.0], device))
        owner = _make_owner(body, device)

        aero = _MockSpringAero(stiffness, target, device)
        dynamics = CompositeBodyJointQ16Dynamics(RigidBodySE3Dynamics(), [], {})
        coupling = PartitionedFreeFlightFSI(aero, dynamics, _MockFreeFlightKinematics())

        result = coupling.advance(owner, dt, time=0.0)

        assert 2 <= result.iterations <= coupling.max_iterations
        assert result.residual <= coupling.tolerance
        assert result.work_error <= 1e-8
        # Exactly one commit for the whole step; owner advanced once.
        assert len(aero.commits) == 1
        assert owner.generation == 1
        assert owner.previous_load is not None
        assert isinstance(owner.previous_load, GeneralizedLoadPacket)

        # Physics: the converged endpoint satisfies the implicit
        # fixed-point equation x* = x0 + k (L - x*) dt^2 / m.
        c = stiffness * dt * dt / mass
        x_star = (x0 + c * target) / (1.0 + c)
        v_star = stiffness * (target - x_star) * dt / mass
        new_body = owner.dynamic_state.body_states["body_0"]
        torch.testing.assert_close(new_body.position_I[0], _vec(x_star, device), rtol=0.0, atol=1e-8)
        torch.testing.assert_close(new_body.linear_velocity_I[0], _vec(v_star, device), rtol=0.0, atol=1e-6)
        # Rotation untouched (force through COM direction keeps moment zero).
        torch.testing.assert_close(
            new_body.quaternion_IB,
            _vec([1.0, 0.0, 0.0, 0.0], device), rtol=0.0, atol=1e-15)
        torch.testing.assert_close(
            new_body.angular_velocity_B, torch.zeros(3, device=device, dtype=DTYPE))

        # True-count semantics (plan 10.2): one proposal + one dynamics
        # advance per trial, one extra replay proposal, one commit.
        counters = coupling.counters
        assert counters.dynamic_proposal_count == result.iterations
        assert counters.aero_proposal_count == result.iterations + 1  # + formal replay
        assert counters.discarded_trial_count == result.iterations - 1
        assert counters.formal_replay_count == 1
        assert counters.commit_count == 1

    @pytest.mark.parametrize("device", DEVICES)
    def test_zero_load_converges_in_one_iteration(self, device):
        body = RigidBodyState.identity("body_0", 2.0, (1.0, 1.0, 1.0), device=device)
        owner = _make_owner(body, device)
        aero = _MockEmptyAero()
        dynamics = CompositeBodyJointQ16Dynamics(RigidBodySE3Dynamics(), [], {})
        coupling = PartitionedFreeFlightFSI(aero, dynamics, _MockFreeFlightKinematics())

        result = coupling.advance(owner, dt=1e-3, time=0.0)

        assert result.iterations == 1
        assert result.residual == 0.0
        assert result.work_error == 0.0
        assert len(aero.commits) == 1
        assert owner.generation == 1
        # Zero wrench from rest: state bit-for-bit unchanged.
        new_body = owner.dynamic_state.body_states["body_0"]
        torch.testing.assert_close(new_body.position_I, body.position_I, rtol=0.0, atol=0.0)
        torch.testing.assert_close(new_body.quaternion_IB, body.quaternion_IB, rtol=0.0, atol=0.0)
        assert coupling.counters.discarded_trial_count == 0
        assert coupling.counters.commit_count == 1

    @pytest.mark.parametrize("device", DEVICES)
    def test_non_convergence_raises_and_never_commits(self, device):
        body = dataclasses.replace(
            RigidBodyState.identity("body_0", 1.0, (1.0, 1.0, 1.0), device=device),
            position_I=_vec([0.5, 0.0, 0.0], device))
        owner = _make_owner(body, device)
        aero = _MockEmptyAero()
        dynamics = _ExpandingDynamics()
        coupling = PartitionedFreeFlightFSI(
            aero, dynamics, _MockFreeFlightKinematics(), max_iterations=3)

        with pytest.raises(RuntimeError, match="did not converge"):
            coupling.advance(owner, dt=1e-3, time=0.0)

        assert len(aero.commits) == 0
        assert coupling.counters.commit_count == 0
        assert coupling.counters.discarded_trial_count == 3
        assert dynamics.calls == 3
        # Owner untouched: trials never leak into the committed state.
        assert owner.generation == 0
        torch.testing.assert_close(
            owner.dynamic_state.body_states["body_0"].position_I,
            _vec([0.5, 0.0, 0.0], device), rtol=0.0, atol=0.0)

    @pytest.mark.parametrize("device", DEVICES)
    def test_work_conjugacy_gate_fails_closed(self, device):
        # Surface power 1 (force along velocity) vs generalized power 0
        # (body at rest): the physics gate must reject the trial.
        class _InconsistentAero(_MockEmptyAero):
            def propose(self, aero_state, frames, dt):
                from types import SimpleNamespace as NS
                force = _vec([[1.0, 0.0, 0.0]], device)
                point = _vec([[1.0, 0.0, 0.0]], device)
                velocity = _vec([[1.0, 0.0, 0.0]], device)
                load = SurfaceLoad.from_panels("wing_0", force, point, velocity)
                proposal = NS(surface_loads=(load,))
                self.proposals.append(proposal)
                return proposal

        body = RigidBodyState.identity("body_0", 1.0, (1.0, 1.0, 1.0), device=device)
        owner = _make_owner(body, device)
        aero = _InconsistentAero()
        dynamics = CompositeBodyJointQ16Dynamics(RigidBodySE3Dynamics(), [], {})
        coupling = PartitionedFreeFlightFSI(aero, dynamics, _MockFreeFlightKinematics())

        with pytest.raises(RuntimeError, match="work-conjugacy gate failed"):
            coupling.advance(owner, dt=1e-3, time=0.0)
        assert len(aero.commits) == 0
        assert owner.generation == 0

    @pytest.mark.parametrize("device", DEVICES)
    def test_relaxed_guess_blends_toward_candidate(self, device):
        # With relaxation 1.0 the guess equals the previous candidate, so
        # convergence is plain fixed-point iteration.
        mass = 1.0
        stiffness = 2.0e3
        dt = 1e-3
        body = dataclasses.replace(
            RigidBodyState.identity("body_0", mass, (1.0, 1.0, 1.0), device=device),
            position_I=_vec([0.2, 0.0, 0.0], device))
        owner = _make_owner(body, device)
        aero = _MockSpringAero(stiffness, 1.0, device)
        dynamics = CompositeBodyJointQ16Dynamics(RigidBodySE3Dynamics(), [], {})
        coupling = PartitionedFreeFlightFSI(
            aero, dynamics, _MockFreeFlightKinematics(), relaxation=1.0)
        result = coupling.advance(owner, dt=dt, time=0.0)
        c = stiffness * dt * dt / mass
        x_star = (0.2 + c) / (1.0 + c)
        torch.testing.assert_close(
            owner.dynamic_state.body_states["body_0"].position_I[0],
            _vec(x_star, device), rtol=0.0, atol=1e-9)
        assert result.residual <= coupling.tolerance


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
