"""Composite dynamics: body + joints + elastic in one advance."""
from __future__ import annotations
from typing import Any
import torch


class CompositeBodyJointQ16Dynamics:
    """Advances body SE(3) + prescribed joints + Q16 elastic in one step.

    The body sees the total wrench from all surfaces (already projected
    via J^T f). Each Q16 wing sees its own elastic generalized force.

    SKELETON for U6: the ``GeneralizedLoadPacket`` contract (plan §7.3,
    one J^T f projection shared by body/joint/elastic loads) is not yet
    defined.  ``propose`` currently reads the body wrench and per-surface
    generalized forces through optional attributes and falls back to zero
    loads; U6 replaces this with the formal packet.
    """
    def __init__(self, body_dynamics, joint_dynamics_list, elastic_adapters: dict):
        self.body = body_dynamics
        self.joints = list(joint_dynamics_list)
        self.elastic = elastic_adapters  # elastic_id → Q16DynamicsAdapter

    def propose(self, committed, loads, dt: float, time: float = 0.0):
        """Advance all subsystems under their respective loads."""
        # TODO(U6): replace with the GeneralizedLoadPacket contract:
        #   body wrench about body COM, joint torques via J_J^T f, and
        #   per-surface elastic generalized forces — all from ONE surface
        #   load projection (plan §7.3), never three independent ones.
        body_committed = getattr(
            committed, "body_states", {}).get(getattr(self.body, "body_id", None)) \
            if hasattr(committed, "body_states") else None
        if body_committed is None and hasattr(committed, "body_states") and committed.body_states:
            body_committed = next(iter(committed.body_states.values()))

        device = (body_committed.position_I.device if body_committed is not None else "cuda:0")
        dtype = (body_committed.position_I.dtype if body_committed is not None else torch.float64)
        body_force = torch.zeros(3, device=device, dtype=dtype)
        body_moment = torch.zeros(3, device=device, dtype=dtype)
        if loads is not None:
            packet_force = getattr(loads, "body_force_I", None)
            packet_moment = getattr(loads, "body_moment_B", None)
            if packet_force is not None:
                body_force = packet_force
            if packet_moment is not None:
                body_moment = packet_moment

        new_states: dict[str, Any] = {"body_states": {}, "joint_states": {}, "elastic_states": {}}

        if body_committed is not None:
            new_states["body_states"][body_committed.body_id] = self.body.propose(
                body_committed, body_force, body_moment, dt)

        # Prescribed joints carry no dynamics: they are pure functions of time.
        for joint in self.joints:
            state = joint.evaluate(time + dt)
            new_states["joint_states"][state.joint_id] = state

        # Elastic subsystems advance through their existing adapters under
        # their per-surface generalized force (TODO(U6): from the packet).
        elastic_loads = getattr(loads, "elastic_generalized_forces", {})
        for elastic_id, adapter in self.elastic.items():
            new_states["elastic_states"][elastic_id] = adapter.propose(
                committed, elastic_loads.get(elastic_id), dt)

        return new_states
