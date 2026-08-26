"""U5: SE(3) body dynamics, prescribed joints, and the moving-root skeleton.

Plan §7/§14 U5: one 6-DOF body owner per aircraft, wings attached through
prescribed joints, Q16 wings with a moving root.  These tests pin the
dynamics-side contracts before U6 composes them into free flight:

- ``RigidBodyState.identity`` defaults and ``validate()`` gates (unit
  quaternion, positive mass, positive-definite inertia);
- ``RigidBodySE3Dynamics.propose`` physics:
  * zero wrench → state unchanged;
  * constant force → semi-implicit quadratic position growth
    x_k = a dt^2 k(k+1)/2;
  * constant moment about a principal axis → linear angular-velocity
    growth and an exactly composed z-rotation quaternion
    Theta_n = (M/I) dt^2 n(n+1)/2;
  * torque-free rotation: I·ω exactly constant for a spherical body;
    |L| and L_z constant in the body frame and L constant in the
    inertial frame for an axisymmetric body;
  * quaternion stays unit after rotating steps;
- ``RigidBodySE3Dynamics.rotation_matrix`` matches the production
  quaternion→matrix convention used by ``prescribed_rigid.py``;
- ``PrescribedJointDynamics`` evaluates a sinusoidal flap law exactly;
- ``CompositeBodyJointQ16Dynamics`` skeleton instantiates and advances
  body + prescribed joint under zero loads without crashing.
"""
from __future__ import annotations

import dataclasses
import math

import pytest
import torch

from fluxvortex.dynamics.composite import CompositeBodyJointQ16Dynamics
from fluxvortex.dynamics.joints import JointState, PrescribedJointDynamics
from fluxvortex.dynamics.rigid_body_se3 import (
    RigidBodySE3Dynamics,
    RigidBodyState,
)
from fluxvortex.state.world import WorldDynamicState

DTYPE = torch.float64

DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda:0")


# ---------------------------------------------------------------------------
# RigidBodyState
# ---------------------------------------------------------------------------

class TestRigidBodyStateIdentity:
    @pytest.mark.parametrize("device", DEVICES)
    def test_identity_defaults(self, device):
        state = RigidBodyState.identity("body_0", 2.5, (1.0, 2.0, 3.0), device=device)
        assert state.body_id == "body_0"
        assert state.position_I.device.type == torch.device(device).type
        for tensor in (state.position_I, state.quaternion_IB,
                       state.linear_velocity_I, state.angular_velocity_B):
            assert tensor.dtype == DTYPE
        for tensor in (state.position_I, state.linear_velocity_I,
                       state.angular_velocity_B):
            assert tensor.shape == (3,)
        assert state.quaternion_IB.shape == (4,)
        assert float(state.mass.item()) == 2.5
        assert state.mass.dtype == DTYPE
        assert torch.equal(state.position_I, torch.zeros(3, device=device, dtype=DTYPE))
        assert torch.equal(
            state.quaternion_IB,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=DTYPE))
        assert torch.equal(state.linear_velocity_I, torch.zeros(3, device=device, dtype=DTYPE))
        assert torch.equal(state.angular_velocity_B, torch.zeros(3, device=device, dtype=DTYPE))
        assert torch.equal(
            state.inertia_B,
            torch.diag(torch.tensor([1.0, 2.0, 3.0], device=device, dtype=DTYPE)))
        state.validate()  # must not raise

    @pytest.mark.parametrize("device", DEVICES)
    def test_identity_validation_passes(self, device):
        RigidBodyState.identity("fuselage", 1.0, (0.5, 0.5, 0.5), device=device).validate()


class TestRigidBodyStateValidation:
    @pytest.mark.parametrize("device", DEVICES)
    def test_non_unit_quaternion_rejected(self, device):
        state = RigidBodyState.identity("body_0", 1.0, (1.0, 1.0, 1.0), device=device)
        bad = dataclasses.replace(state, quaternion_IB=1.1 * state.quaternion_IB)
        with pytest.raises(ValueError, match="quaternion not unit"):
            bad.validate()

    @pytest.mark.parametrize("device", DEVICES)
    def test_non_finite_quaternion_rejected(self, device):
        state = RigidBodyState.identity("body_0", 1.0, (1.0, 1.0, 1.0), device=device)
        quat = state.quaternion_IB.clone()
        quat[0] = float("nan")
        with pytest.raises(ValueError, match="quaternion non-finite"):
            dataclasses.replace(state, quaternion_IB=quat).validate()

    @pytest.mark.parametrize("device", DEVICES)
    def test_negative_mass_rejected(self, device):
        state = RigidBodyState.identity("body_0", 1.0, (1.0, 1.0, 1.0), device=device)
        bad = dataclasses.replace(
            state, mass=torch.tensor(-2.0, device=device, dtype=DTYPE))
        with pytest.raises(ValueError, match="mass not positive"):
            bad.validate()

    @pytest.mark.parametrize("device", DEVICES)
    def test_non_pd_inertia_rejected(self, device):
        state = RigidBodyState.identity("body_0", 1.0, (1.0, 1.0, 1.0), device=device)
        for diag in ((1.0, 1.0, -1.0), (1.0, 0.0, 1.0)):
            bad = dataclasses.replace(
                state,
                inertia_B=torch.diag(torch.tensor(diag, device=device, dtype=DTYPE)))
            with pytest.raises(ValueError, match="inertia not positive definite"):
                bad.validate()


# ---------------------------------------------------------------------------
# RigidBodySE3Dynamics.propose
# ---------------------------------------------------------------------------

class TestSE3Propose:
    @pytest.mark.parametrize("device", DEVICES)
    def test_zero_wrench_leaves_state_unchanged(self, device):
        dyn = RigidBodySE3Dynamics()
        state = RigidBodyState.identity("body_0", 3.0, (1.0, 2.0, 3.0), device=device)
        zero_f = torch.zeros(3, device=device, dtype=DTYPE)
        zero_m = torch.zeros(3, device=device, dtype=DTYPE)
        for _ in range(5):
            state = dyn.propose(state, zero_f, zero_m, dt=1e-3)
        assert torch.equal(state.position_I, torch.zeros(3, device=device, dtype=DTYPE))
        assert torch.equal(state.linear_velocity_I, torch.zeros(3, device=device, dtype=DTYPE))
        assert torch.equal(state.angular_velocity_B, torch.zeros(3, device=device, dtype=DTYPE))
        assert torch.equal(
            state.quaternion_IB,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=DTYPE))

    @pytest.mark.parametrize("device", DEVICES)
    def test_constant_force_quadratic_position(self, device):
        dyn = RigidBodySE3Dynamics()
        mass = 2.0
        state = RigidBodyState.identity("body_0", mass, (1.0, 1.0, 1.0), device=device)
        force = torch.tensor([0.0, 0.0, 4.0], device=device, dtype=DTYPE)
        zero_m = torch.zeros(3, device=device, dtype=DTYPE)
        dt = 1e-3
        accel = float((force / state.mass)[2].item())

        positions = []
        for _ in range(3):
            state = dyn.propose(state, force, zero_m, dt=dt)
            positions.append(state.position_I.clone())

        # Semi-implicit from rest: x_k = a dt^2 k(k+1)/2 (k = 1, 2, 3).
        for k, pos in enumerate(positions, start=1):
            expected_z = accel * dt * dt * k * (k + 1) / 2.0
            torch.testing.assert_close(
                pos[2], torch.tensor(expected_z, device=device, dtype=DTYPE))
            assert float(pos[0].item()) == 0.0
            assert float(pos[1].item()) == 0.0
        # Velocity grows linearly: v_k = k a dt.
        torch.testing.assert_close(
            state.linear_velocity_I[2], torch.tensor(3 * accel * dt, device=device, dtype=DTYPE))

    @pytest.mark.parametrize("device", DEVICES)
    def test_constant_moment_linear_angular_velocity(self, device):
        dyn = RigidBodySE3Dynamics()
        I_z = 3.0
        state = RigidBodyState.identity("body_0", 1.0, (1.0, 2.0, I_z), device=device)
        zero_f = torch.zeros(3, device=device, dtype=DTYPE)
        moment = torch.tensor([0.0, 0.0, 0.6], device=device, dtype=DTYPE)
        dt = 1e-3
        n_steps = 6

        for k in range(1, n_steps + 1):
            state = dyn.propose(state, zero_f, moment, dt=dt)
            # ω stays on the z principal axis and grows linearly: ω_k = k (M/I) dt.
            assert float(state.angular_velocity_B[0].item()) == 0.0
            assert float(state.angular_velocity_B[1].item()) == 0.0
            torch.testing.assert_close(
                state.angular_velocity_B[2],
                torch.tensor(k * (0.6 / I_z) * dt, device=device, dtype=DTYPE))

        # All incremental rotations are about z, so they compose exactly:
        # Θ_n = (M/I) dt^2 n(n+1)/2 and q = (cos(Θ/2), 0, 0, sin(Θ/2)).
        theta = (0.6 / I_z) * dt * dt * n_steps * (n_steps + 1) / 2.0
        expected = torch.tensor(
            [math.cos(theta / 2), 0.0, 0.0, math.sin(theta / 2)],
            device=device, dtype=DTYPE)
        torch.testing.assert_close(state.quaternion_IB, expected, rtol=1e-12, atol=1e-14)

    @pytest.mark.parametrize("device", DEVICES)
    def test_free_rotation_spherical_body_constant_body_omega(self, device):
        # Spherical inertia: the gyroscopic term ω×(Iω) vanishes identically,
        # so I·ω (and ω) is exactly constant in the body frame.
        dyn = RigidBodySE3Dynamics()
        state = RigidBodyState.identity("body_0", 1.0, (2.0, 2.0, 2.0), device=device)
        omega0 = torch.tensor([0.3, -0.7, 1.1], device=device, dtype=DTYPE)
        state = dataclasses.replace(state, angular_velocity_B=omega0.clone())
        zero_f = torch.zeros(3, device=device, dtype=DTYPE)
        zero_m = torch.zeros(3, device=device, dtype=DTYPE)
        L0 = state.inertia_B @ omega0

        for _ in range(200):
            state = dyn.propose(state, zero_f, zero_m, dt=1e-3)
        torch.testing.assert_close(state.angular_velocity_B, omega0)
        torch.testing.assert_close(state.inertia_B @ state.angular_velocity_B, L0)

    @pytest.mark.parametrize("device", DEVICES)
    def test_free_rotation_axisymmetric_conserves_angular_momentum(self, device):
        # Axisymmetric body (Ix = Iy != Iz) with tilted ω: in the body frame
        # L precesses around the symmetry axis, so only |L| and L_z are
        # constant there; the full vector L is conserved in the INERTIAL
        # frame (L_I = R_IB · I ω).  The semi-implicit scheme is first
        # order, so |L| / L_I drift over a fixed horizon must (a) stay
        # small and (b) halve when dt is halved, while L_z is preserved
        # exactly (the gyroscopic term ω×(Iω) never has a z component for
        # a diagonal, axisymmetric inertia).

        def drift(dt: float, n_steps: int) -> tuple[float, float, float]:
            dyn = RigidBodySE3Dynamics()
            state = RigidBodyState.identity("body_0", 1.0, (2.0, 2.0, 1.0), device=device)
            omega0 = torch.tensor([0.8, 0.0, 1.5], device=device, dtype=DTYPE)
            state = dataclasses.replace(state, angular_velocity_B=omega0.clone())
            zero_f = torch.zeros(3, device=device, dtype=DTYPE)
            zero_m = torch.zeros(3, device=device, dtype=DTYPE)
            L_body0 = state.inertia_B @ omega0
            L_inertial0 = dyn.rotation_matrix(state.quaternion_IB) @ L_body0
            for _ in range(n_steps):  # fixed horizon t = n * dt = 0.5 s
                state = dyn.propose(state, zero_f, zero_m, dt=dt)
            L_body = state.inertia_B @ state.angular_velocity_B
            L_inertial = dyn.rotation_matrix(state.quaternion_IB) @ L_body
            norm_drift = float(
                (torch.linalg.vector_norm(L_body) - torch.linalg.vector_norm(L_body0)).item())
            z_drift = float((L_body[2] - L_body0[2]).abs().item())
            inertial_drift = float(torch.linalg.vector_norm(L_inertial - L_inertial0).item())
            return norm_drift, z_drift, inertial_drift

        norm_coarse, z_coarse, inertial_coarse = drift(dt=5.0e-4, n_steps=1000)
        norm_fine, z_fine, inertial_fine = drift(dt=2.5e-4, n_steps=2000)

        # Symmetry-axis angular momentum: conserved exactly at both steps.
        assert z_coarse == 0.0 or abs(z_coarse) < 1e-12
        assert z_fine == 0.0 or abs(z_fine) < 1e-12
        # |L| and inertial L: small drift at the fine step, halved from coarse
        # (first-order scheme, fixed horizon).
        L_scale = float(torch.linalg.vector_norm(
            torch.tensor([1.6, 0.0, 1.5], dtype=DTYPE)).item())
        assert norm_fine < 1e-4 * L_scale
        assert inertial_fine < 1e-4 * L_scale
        assert abs(norm_coarse / norm_fine - 2.0) < 0.5
        assert abs(inertial_coarse / inertial_fine - 2.0) < 0.5

    @pytest.mark.parametrize("device", DEVICES)
    def test_quaternion_stays_unit_under_rotation(self, device):
        dyn = RigidBodySE3Dynamics()
        state = RigidBodyState.identity("body_0", 1.0, (1.0, 1.0, 1.0), device=device)
        zero_f = torch.zeros(3, device=device, dtype=DTYPE)
        moment = torch.tensor([0.05, -0.02, 0.1], device=device, dtype=DTYPE)
        for _ in range(500):
            state = dyn.propose(state, zero_f, moment, dt=1e-3)
            norm = float(torch.linalg.vector_norm(state.quaternion_IB).item())
            assert abs(norm - 1.0) < 1e-12
        # Rotation actually happened (not the identity quaternion any more).
        assert abs(float(state.quaternion_IB[1].item())) > 1e-3


# ---------------------------------------------------------------------------
# rotation_matrix convention
# ---------------------------------------------------------------------------

class TestRotationMatrix:
    @pytest.mark.parametrize("device", DEVICES)
    def test_matches_prescribed_rigid_convention(self, device):
        # Same [w,x,y,z] → R_IB formula as PrescribedRigidSurfaceKinematics.
        from fluxvortex.kinematics.prescribed_rigid import (
            PrescribedRigidSurfaceKinematics,
        )
        dyn = RigidBodySE3Dynamics()
        quat = torch.tensor(
            [math.cos(0.3), 0.0, 0.0, math.sin(0.3)], device=device, dtype=DTYPE)
        expected = PrescribedRigidSurfaceKinematics._quaternion_to_matrix(quat)
        torch.testing.assert_close(dyn.rotation_matrix(quat), expected)

    @pytest.mark.parametrize("device", DEVICES)
    def test_z_rotation_matrix_rotates_body_to_inertial(self, device):
        dyn = RigidBodySE3Dynamics()
        theta = math.pi / 2
        quat = torch.tensor(
            [math.cos(theta / 2), 0.0, 0.0, math.sin(theta / 2)],
            device=device, dtype=DTYPE)
        R = dyn.rotation_matrix(quat)
        body_x = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=DTYPE)
        rotated = R @ body_x
        torch.testing.assert_close(
            rotated, torch.tensor([0.0, 1.0, 0.0], device=device, dtype=DTYPE),
            rtol=0.0, atol=1e-15)
        # Orthonormal with determinant +1.
        torch.testing.assert_close(
            R @ R.T, torch.eye(3, device=device, dtype=DTYPE), rtol=0.0, atol=1e-15)
        torch.testing.assert_close(
            torch.linalg.det(R), torch.tensor(1.0, device=device, dtype=DTYPE),
            rtol=0.0, atol=1e-15)


# ---------------------------------------------------------------------------
# PrescribedJointDynamics
# ---------------------------------------------------------------------------

class TestPrescribedJointDynamics:
    @pytest.mark.parametrize("device", DEVICES)
    def test_sinusoidal_flap_law(self, device):
        amplitude = math.radians(25.0)
        frequency = 2.0
        omega = 2.0 * math.pi * frequency

        def flap_law(t: float):
            coords = torch.tensor([amplitude * math.sin(omega * t)], device=device, dtype=DTYPE)
            rates = torch.tensor(
                [amplitude * omega * math.cos(omega * t)], device=device, dtype=DTYPE)
            return coords, rates

        joint = PrescribedJointDynamics("right_flap", "body_0", flap_law)
        for t, expected_coord in (
            (0.0, 0.0),
            (0.125, amplitude),        # quarter period: sin(π/2) = 1
            (0.25, 0.0),               # half period
            (0.375, -amplitude),
        ):
            state = joint.evaluate(t)
            assert isinstance(state, JointState)
            assert state.joint_id == "right_flap"
            assert state.parent_body_id == "body_0"
            assert state.prescribed is True
            assert state.coordinates.shape == (1,)
            assert state.rates.shape == (1,)
            expected_rate = amplitude * omega * math.cos(omega * t)
            torch.testing.assert_close(
                state.coordinates,
                torch.tensor([expected_coord], device=device, dtype=DTYPE))
            torch.testing.assert_close(
                state.rates,
                torch.tensor([expected_rate], device=device, dtype=DTYPE))

    @pytest.mark.parametrize("device", DEVICES)
    def test_two_dof_flap_pitch_law(self, device):
        def law(t: float):
            return (
                torch.tensor([0.1 * t, math.radians(5.0)], device=device, dtype=DTYPE),
                torch.tensor([0.1, 0.0], device=device, dtype=DTYPE),
            )

        joint = PrescribedJointDynamics("left_wing_joint", "body_0", law)
        state = joint.evaluate(2.0)
        assert state.coordinates.shape == (2,)
        torch.testing.assert_close(
            state.coordinates, torch.tensor([0.2, math.radians(5.0)], device=device, dtype=DTYPE))
        torch.testing.assert_close(
            state.rates, torch.tensor([0.1, 0.0], device=device, dtype=DTYPE))


# ---------------------------------------------------------------------------
# CompositeBodyJointQ16Dynamics skeleton
# ---------------------------------------------------------------------------

class _DummyElasticAdapter:
    """Records propose calls; stands in for a Q16DynamicsAdapter."""

    def __init__(self) -> None:
        self.calls = []

    def predict(self, committed, dt):
        return None

    def propose(self, committed, loads, dt):
        self.calls.append((loads, dt))
        return "elastic_proposed"


class TestCompositeSkeleton:
    @pytest.mark.parametrize("device", DEVICES)
    def test_instantiation_and_zero_load_advance(self, device):
        body = RigidBodySE3Dynamics()

        def flap_law(t: float):
            return (
                torch.tensor([0.1 * t], device=device, dtype=DTYPE),
                torch.tensor([0.1], device=device, dtype=DTYPE),
            )

        joint = PrescribedJointDynamics("right_flap", "body_0", flap_law)
        elastic = _DummyElasticAdapter()
        composite = CompositeBodyJointQ16Dynamics(body, [joint], {"wing_0": elastic})
        assert composite.body is body
        assert composite.joints == [joint]
        assert composite.elastic == {"wing_0": elastic}

        body_state = RigidBodyState.identity("body_0", 2.0, (1.0, 1.0, 1.0), device=device)
        committed = WorldDynamicState(
            elastic_states={"wing_0": object()},
            body_states={"body_0": body_state},
            joint_states={},
        )
        advanced = composite.propose(committed, None, dt=1e-3, time=0.5)

        # Body advanced one zero-wrench step from rest: unchanged pose.
        new_body = advanced["body_states"]["body_0"]
        assert isinstance(new_body, RigidBodyState)
        torch.testing.assert_close(new_body.position_I, body_state.position_I)
        torch.testing.assert_close(new_body.quaternion_IB, body_state.quaternion_IB)
        # Prescribed joint evaluated at t + dt.
        new_joint = advanced["joint_states"]["right_flap"]
        torch.testing.assert_close(
            new_joint.coordinates, torch.tensor([0.1 * 0.501], device=device, dtype=DTYPE))
        torch.testing.assert_close(
            new_joint.rates, torch.tensor([0.1], device=device, dtype=DTYPE))
        # Elastic adapter was driven exactly once with its (zero) load.
        assert len(elastic.calls) == 1
        assert elastic.calls[0][1] == 1e-3


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
