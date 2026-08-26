"""BodyJointQ16Kinematics: surface frame from body pose + joint angles + Q16 deformation.

Implements plan §7.2: x_I = x_B + R_IB · T_BJ(q_J) · (X_S + u_e)

The velocity chain is non-negotiable (plan §7.2): the surface velocity
seen by the aero solver must contain ALL FOUR terms — body translation,
body angular rate cross product (ω × r), joint rate, and the Q16 elastic
rate.  Adding the body pose to positions while dropping any velocity
term corrupts the no-penetration condition and the added-mass loads.
"""
from __future__ import annotations
from typing import Any, Callable
import torch


class BodyJointQ16Kinematics:
    """Chain: body SE(3) pose → joint transform → Q16 surface deformation → SurfaceFrame."""

    def __init__(self, q16_surface_adapter, joint_transform, *, surface_id, body_id,
                 joint_velocity: Callable[[Any], tuple[torch.Tensor, torch.Tensor]] | None = None):
        """
        Parameters
        ----------
        q16_surface_adapter:
            Q16SurfaceFrameAdapter-like; ``evaluate(q, qd)`` returns the
            LOCAL surface frame (surface-reference components).
        joint_transform:
            Callable ``(JointState) -> (p_BJ, R_BJ)``: joint origin and
            rotation (surface reference → body frame).
        joint_velocity:
            Optional callable ``(JointState) -> (pdot_BJ, omega_JB)``:
            joint origin velocity and joint angular velocity in the body
            frame.  REQUIRED for moving joints — when the joint state
            reports non-zero rates and no mapping is installed, the
            evaluation fails closed instead of silently dropping the
            joint-rate velocity term.
        """
        self._q16 = q16_surface_adapter       # Q16SurfaceFrameAdapter (local frame)
        self._joint = joint_transform          # Callable (JointState) → (pos_J, R_BJ)
        self._joint_velocity = joint_velocity
        self.surface_id = surface_id
        self.body_id = body_id

    def evaluate(self, body_state, joint_state, elastic_state, time: float | None = None):
        """Full kinematic chain §7.2: x_I = x_B + R_IB·T_BJ(q_J)·(X_S + u_e).

        Velocities include: body translation + body rotation (ω×r) +
        joint rates + elastic velocity (plan §7.2's non-negotiable
        requirement).  ``time`` is accepted for protocol symmetry; the
        joint state already carries the clock's coordinates and rates.
        """
        from ..dynamics.rigid_body_se3 import RigidBodySE3Dynamics
        R_IB = RigidBodySE3Dynamics().rotation_matrix(body_state.quaternion_IB)

        # Joint transform: position offset and rotation in body frame
        p_BJ, R_BJ = self._joint(joint_state)

        # Joint rate terms (body frame): origin velocity + angular velocity.
        rates = getattr(joint_state, "rates", None)
        device = body_state.position_I.device
        dtype = body_state.position_I.dtype
        zero3 = torch.zeros(3, device=device, dtype=dtype)
        if self._joint_velocity is not None:
            pdot_BJ, omega_JB = self._joint_velocity(joint_state)
            pdot_BJ = pdot_BJ.to(device=device, dtype=dtype)
            omega_JB = omega_JB.to(device=device, dtype=dtype)
        else:
            if rates is not None and bool((rates != 0).any()):
                raise ValueError(
                    "joint state reports non-zero rates but no joint_velocity "
                    "mapping was provided: refusing to drop the joint-rate "
                    "velocity term (plan 7.2)")
            pdot_BJ, omega_JB = zero3, zero3.clone()

        # Combined body→surface transform: R_BS, p_BS with S the surface
        # reference frame, so x_I = x_com + R_IB (p_BS + R_BS x_S).
        R_BS = R_BJ                              # body → surface reference
        p_BS = p_BJ                              # surface reference origin in body frame
        R_IS = R_IB @ R_BS
        p_IS = body_state.position_I + R_IB @ p_BS

        # Local Q16 frame (in surface reference frame)
        local = self._q16.evaluate(elastic_state.q, elastic_state.qd)

        # Transform to inertial frame: x_I = R_IS x_S + p_IS
        def to_I(points: torch.Tensor) -> torch.Tensor:
            return (points @ R_IS.T) + p_IS

        def vel_I(points: torch.Tensor, local_vels: torch.Tensor) -> torch.Tensor:
            """v_I = v_com + ω_I × (x_I - x_com) + R_IB (pdot_BJ + ω_JB × (R_BJ x_S) + R_BJ u_S)."""
            x_B = (points @ R_BS.T) + p_BS          # body-frame position
            rel_I = x_B @ R_IB.T                    # = x_I - x_com
            # ω_I = R_IB · ω_B (body angular velocity to inertial frame)
            omega_I = R_IB @ body_state.angular_velocity_B
            rigid_vel = body_state.linear_velocity_I + torch.cross(
                omega_I.expand_as(points), rel_I, dim=-1)
            # Joint-rate block, expressed in the body frame:
            #   pdot_BJ + ω_JB × (R_BJ x_S)   with R_BJ x_S = x_B - p_BS
            joint_vel_B = pdot_BJ.expand_as(points) + torch.cross(
                omega_JB.expand_as(points), x_B - p_BS, dim=-1)
            return rigid_vel + (joint_vel_B + local_vels @ R_BS.T) @ R_IB.T

        return type(local)(
            surface_id=self.surface_id,
            body_id=self.body_id,
            panel_rings_I=to_I(local.panel_rings_I),
            panel_ring_velocity_I=vel_I(
                local.panel_rings_I.reshape(-1, 3),
                local.panel_ring_velocity_I.reshape(-1, 3),
            ).reshape(local.panel_rings_I.shape),
            collocation_I=to_I(local.collocation_I),
            collocation_velocity_I=vel_I(local.collocation_I, local.collocation_velocity_I),
            normals_I=local.normals_I @ R_IS.T,
            areas=local.areas,
            leading_edge_I=to_I(local.leading_edge_I),
            trailing_edge_I=to_I(local.trailing_edge_I),
            leading_velocity_I=vel_I(local.leading_edge_I, local.leading_velocity_I),
            trailing_velocity_I=vel_I(local.trailing_edge_I, local.trailing_velocity_I),
            chordwise_panels=local.chordwise_panels,
            spanwise_panels=local.spanwise_panels,
            topology_digest=local.topology_digest,
        )
