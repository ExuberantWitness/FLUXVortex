"""Prescribed joint dynamics for flapping/pitching wings."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import torch


@dataclass(frozen=True)
class JointState:
    joint_id: str
    parent_body_id: str
    coordinates: torch.Tensor   # (n_dof,) — flap angle, pitch angle
    rates: torch.Tensor         # (n_dof,)
    prescribed: bool = True


class PrescribedJointDynamics:
    """Evaluates joint coordinates from a prescribed motion law."""
    def __init__(self, joint_id: str, parent_body_id: str,
                 motion_law: Callable[[float], tuple[torch.Tensor, torch.Tensor]]):
        self.joint_id = joint_id
        self.parent_body_id = parent_body_id
        self._law = motion_law

    def evaluate(self, time: float) -> JointState:
        coords, rates = self._law(time)
        return JointState(
            joint_id=self.joint_id,
            parent_body_id=self.parent_body_id,
            coordinates=coords,
            rates=rates,
            prescribed=True,
        )
