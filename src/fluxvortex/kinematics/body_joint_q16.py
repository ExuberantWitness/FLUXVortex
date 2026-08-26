"""BodyJointQ16Kinematics: surface frame from body pose + joint angles + Q16 deformation.

Implements plan §7.2: x_I = x_B + R_IB · T_BJ(q_J) · (X_S + u_e)
"""
from __future__ import annotations
from typing import Any
import torch


class BodyJointQ16Kinematics:
    """Chain: body SE(3) pose → joint transform → Q16 surface deformation → SurfaceFrame."""

    def __init__(self, q16_surface_adapter, joint_transform, *, surface_id, body_id):
        self._q16 = q16_surface_adapter       # Q16SurfaceFrameAdapter (local frame)
        self._joint = joint_transform          # Callable (JointState) → (pos_J, R_BJ)
        self.surface_id = surface_id
        self.body_id = body_id

    def evaluate(self, body_state, joint_state, elastic_state) -> Any:
        """Full kinematic chain for one wing.

        U5 SKELETON: only the wiring below is established.  The returned
        frame is still the LOCAL Q16 frame — the position chain
        (joint offset + rotation, then body rotation + translation) and the
        full velocity chain of plan §7.2 (body translation, ω×r arm, joint
        rate, Q16 elastic rate) are U6 work.  Returning the local frame
        untransformed keeps U5 honest: no silently missing velocity terms.
        """
        from ..dynamics.rigid_body_se3 import RigidBodySE3Dynamics
        R_IB = RigidBodySE3Dynamics().rotation_matrix(body_state.quaternion_IB)

        # Joint transform: position offset and rotation in body frame
        joint_pos_B, joint_R_BJ = self._joint(joint_state)

        # Surface in body frame: joint transform applied to local Q16 geometry
        local_frame = self._q16.evaluate(elastic_state.q, elastic_state.qd)

        # Transform local → body → inertial
        def to_body(points):
            return (points @ joint_R_BJ.T) + joint_pos_B
        def to_inertial(points):
            return (points @ R_IB.T) + body_state.position_I

        # TODO(U6): apply to_body/to_inertial to every geometry field and add
        # the velocity chain (body translation + ω_I×r + joint rate + elastic
        # rate) before handing the frame to the aero solver (plan §7.2).
        return local_frame
