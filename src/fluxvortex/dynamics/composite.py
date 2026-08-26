"""Composite dynamics: body + joints + elastic in one advance."""
from __future__ import annotations
from typing import Any
import torch

from .rigid_body_se3 import RigidBodySE3Dynamics


class CompositeBodyJointQ16Dynamics:
    """Advances body SE(3) + prescribed joints + Q16 elastic in one step.

    The body sees the total wrench from all surfaces (already projected
    via J^T f into a ``GeneralizedLoadPacket``, plan §7.3).  Each Q16
    wing sees its own elastic generalized force from the SAME surface
    forces.  Prescribed joints are pure functions of time: they carry no
    state to integrate, and their torques (if any) are bookkeeping on
    the packet, not inputs to an ODE.
    """
    def __init__(self, body_dynamics, joint_dynamics_list, elastic_adapters: dict):
        self.body = body_dynamics
        self.joints = list(joint_dynamics_list)
        self.elastic = elastic_adapters  # elastic_id → Q16DynamicsAdapter

    def propose(self, committed, loads, dt: float, time: float = 0.0):
        """Advance all subsystems under their respective loads.

        ``loads`` follows the U6 ``GeneralizedLoadPacket`` contract:
        one J^T f projection producing the body wrench about the body
        COM (inertial components), joint torques, and per-surface
        elastic generalized forces — never three independent surface
        force evaluations.  ``loads=None`` (U5 legacy tests) means zero
        load on every subsystem.
        """
        body_states = dict(getattr(committed, "body_states", None) or {})

        packet_wrenches = getattr(loads, "body_wrenches_I", None) if loads is not None else None

        new_states: dict[str, Any] = {"body_states": {}, "joint_states": {}, "elastic_states": {}}

        for body_id, body_committed in body_states.items():
            device = body_committed.position_I.device
            dtype = body_committed.position_I.dtype
            force_I = torch.zeros(3, device=device, dtype=dtype)
            moment_I = torch.zeros(3, device=device, dtype=dtype)
            moment_is_body_frame = False
            if packet_wrenches:
                wrench = packet_wrenches.get(body_id) or {}
                if wrench.get("force_I") is not None:
                    force_I = wrench["force_I"].to(device=device, dtype=dtype)
                if wrench.get("moment_I") is not None:
                    moment_I = wrench["moment_I"].to(device=device, dtype=dtype)
            else:
                # U5-era duck-typed fallbacks (kept so old producers still run).
                packet_force = getattr(loads, "body_force_I", None) if loads is not None else None
                packet_moment = getattr(loads, "body_moment_B", None) if loads is not None else None
                if packet_force is not None:
                    force_I = packet_force.to(device=device, dtype=dtype)
                if packet_moment is not None:
                    # Legacy producers already express the moment in body frame.
                    moment_I = packet_moment.to(device=device, dtype=dtype)
                    moment_is_body_frame = True

            if moment_is_body_frame:
                moment_B = moment_I
            else:
                # Packet wrench is about the body COM in INERTIAL components;
                # RigidBodySE3Dynamics.propose consumes body-frame moments:
                # M_B = R_IB^T . M_I.
                R_IB = RigidBodySE3Dynamics().rotation_matrix(body_committed.quaternion_IB)
                moment_B = R_IB.T @ moment_I
            new_states["body_states"][body_id] = self.body.propose(
                body_committed, force_I, moment_B, dt)

        # Prescribed joints carry no dynamics: they are pure functions of time.
        for joint in self.joints:
            state = joint.evaluate(time + dt)
            new_states["joint_states"][state.joint_id] = state

        # Elastic subsystems advance through their existing adapters under
        # their per-surface generalized force from the SAME packet.
        elastic_loads = getattr(loads, "elastic_forces", None) if loads is not None else None
        if elastic_loads is None:
            elastic_loads = getattr(loads, "elastic_generalized_forces", {}) if loads is not None else {}
        for elastic_id, adapter in self.elastic.items():
            new_states["elastic_states"][elastic_id] = adapter.propose(
                committed, elastic_loads.get(elastic_id), dt)

        return new_states
