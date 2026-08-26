"""RigidBodySE3Dynamics: quaternion-based 6-DOF rigid body on CUDA float64.

Integrates using an exponential-map semi-implicit scheme (plan §7.4):
- Translation: implicit midpoint on position/velocity
- Rotation: quaternion multiplication with exponential map of angular velocity
- Quaternion re-normalized after every accepted step
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import math
import torch


@dataclass(frozen=True)
class RigidBodyState:
    body_id: str
    position_I: torch.Tensor       # (3,)
    quaternion_IB: torch.Tensor    # (4,) unit [w,x,y,z]
    linear_velocity_I: torch.Tensor   # (3,)
    angular_velocity_B: torch.Tensor  # (3,) in body frame
    mass: torch.Tensor             # scalar
    inertia_B: torch.Tensor        # (3,3) in body frame, about COM

    def __post_init__(self):
        # Validation happens in the factory; this is a frozen dataclass
        pass

    @staticmethod
    def identity(body_id: str, mass: float, inertia_diag: tuple[float, float, float],
                 device: str = "cuda:0") -> "RigidBodyState":
        return RigidBodyState(
            body_id=body_id,
            position_I=torch.zeros(3, device=device, dtype=torch.float64),
            quaternion_IB=torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=torch.float64),
            linear_velocity_I=torch.zeros(3, device=device, dtype=torch.float64),
            angular_velocity_B=torch.zeros(3, device=device, dtype=torch.float64),
            mass=torch.tensor(mass, device=device, dtype=torch.float64),
            inertia_B=torch.diag(torch.tensor(inertia_diag, device=device, dtype=torch.float64)),
        )

    def validate(self) -> None:
        if not bool(torch.isfinite(self.quaternion_IB).all()):
            raise ValueError("quaternion non-finite")
        norm = float(torch.linalg.vector_norm(self.quaternion_IB).item())
        if abs(norm - 1.0) > 1e-10:
            raise ValueError(f"quaternion not unit: |q|={norm}")
        eig = torch.linalg.eigvalsh(self.inertia_B)
        if not bool((eig > 0).all()):
            raise ValueError("inertia not positive definite")
        if float(self.mass.item()) <= 0:
            raise ValueError("mass not positive")


class RigidBodySE3Dynamics:
    """Semi-implicit 6-DOF integrator with exponential-map rotation."""

    def propose(self, committed: RigidBodyState,
                total_force_I: torch.Tensor,
                total_moment_B: torch.Tensor,
                dt: float) -> RigidBodyState:
        """Advance one step under total force (inertial frame) and moment (body frame)."""
        # Semi-implicit: update velocities first, then positions
        new_lin_vel = committed.linear_velocity_I + (total_force_I / committed.mass) * dt
        # Euler's equation in body frame: I*ω̇ + ω×(I*ω) = M
        Iw = committed.inertia_B @ committed.angular_velocity_B
        gyroscopic = torch.cross(committed.angular_velocity_B, Iw, dim=0)
        ang_accel = torch.linalg.solve(
            committed.inertia_B, total_moment_B - gyroscopic)
        new_ang_vel = committed.angular_velocity_B + ang_accel * dt

        # Position update (inertial frame)
        new_pos = committed.position_I + new_lin_vel * dt

        # Rotation update via exponential map
        speed = float(torch.linalg.vector_norm(new_ang_vel).item())
        angle = speed * dt
        if angle > 1e-14:
            axis = new_ang_vel / speed
            half = angle / 2.0
            dq = torch.cat((
                math.cos(half) * torch.ones(1, device=axis.device, dtype=torch.float64),
                math.sin(half) * axis,
            ))
            # q_new = q_old ⊗ dq (body-frame angular velocity → post-multiply)
            w0, x0, y0, z0 = committed.quaternion_IB.unbind()
            w1, x1, y1, z1 = dq.unbind()
            new_quat = torch.stack([
                w0*w1 - x0*x1 - y0*y1 - z0*z1,
                w0*x1 + x0*w1 + y0*z1 - z0*y1,
                w0*y1 - x0*z1 + y0*w1 + z0*x1,
                w0*z1 + x0*y1 - y0*x1 + z0*w1,
            ])
            new_quat = new_quat / torch.linalg.vector_norm(new_quat)
        else:
            new_quat = committed.quaternion_IB

        return RigidBodyState(
            body_id=committed.body_id,
            position_I=new_pos,
            quaternion_IB=new_quat,
            linear_velocity_I=new_lin_vel,
            angular_velocity_B=new_ang_vel,
            mass=committed.mass,
            inertia_B=committed.inertia_B,
        )

    def rotation_matrix(self, quaternion: torch.Tensor) -> torch.Tensor:
        """Quaternion → rotation matrix R_IB (rotates body→inertial)."""
        w, x, y, z = quaternion.unbind()
        return torch.stack([
            torch.stack([1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)]),
            torch.stack([2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)]),
            torch.stack([2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]),
        ])
