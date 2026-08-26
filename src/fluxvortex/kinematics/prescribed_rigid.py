"""PrescribedRigidSurfaceKinematics: rigid surface with given motion law.

U3 (plan §11.3/§14): rigid paper cases (Yang 2025 four-bar mechanism,
Izraelevitz-Scherer heave/pitch flapper, ...) drive the unified aerodynamics
through the same ``SurfaceFrame`` contract as the flexible Q16 path — the
surface is simply prescribed instead of structurally computed.
"""
from __future__ import annotations

from typing import Any, Callable

import torch

from .frames import SurfaceFrame


class PrescribedRigidSurfaceKinematics:
    """Evaluates a rigid surface's SurfaceFrame from a prescribed motion law.

    The motion law is a callable ``(time) -> (position (3,), quaternion (4,),
    linear_velocity (3,), angular_velocity (3,))``.  The surface's reference
    geometry is rotated/translated to the current pose.  Panel point
    velocities are the rigid-body field ``v + omega x r`` evaluated at every
    geometry node; normals rotate with the body while panel areas are
    unchanged (rigid motion preserves area).
    """

    def __init__(
        self,
        reference_geometry: SurfaceFrame,
        motion_law: Callable[[float], Any],
        *,
        surface_id: str,
        body_id: str,
    ) -> None:
        # reference_geometry: SurfaceFrame at the reference pose
        self._ref = reference_geometry
        self._law = motion_law
        self.surface_id = surface_id
        self.body_id = body_id

    @property
    def surface_ids(self) -> tuple[str, ...]:
        return (self.surface_id,)

    def evaluate(self, time: float) -> SurfaceFrame:
        pos, quat, lin_vel, ang_vel = self._law(time)
        pos = self._coerce(pos)
        quat = self._coerce(quat)
        lin_vel = self._coerce(lin_vel)
        ang_vel = self._coerce(ang_vel)
        R = self._quaternion_to_matrix(quat)

        def transform_points(points: torch.Tensor) -> torch.Tensor:
            return (points @ R.T) + pos

        def transform_vectors(vectors: torch.Tensor) -> torch.Tensor:
            return vectors @ R.T

        # Surface velocities = rigid body velocity + ω × r, with r the point
        # position in the CURRENT pose relative to the body origin.  Since
        # x(t) - pos = R x_ref, the field is analytic in the reference points
        # (no second transform pass); note ω × x_ref would be wrong — the
        # arm must be the rotated one, ω × (R x_ref).
        def point_velocity(points: torch.Tensor) -> torch.Tensor:
            rel = points @ R.T
            return lin_vel + torch.cross(ang_vel.expand_as(rel), rel, dim=-1)

        ref = self._ref
        return SurfaceFrame(
            surface_id=self.surface_id,
            body_id=self.body_id,
            panel_rings_I=transform_points(ref.panel_rings_I),
            panel_ring_velocity_I=point_velocity(
                ref.panel_rings_I.reshape(-1, 3)
            ).reshape_as(ref.panel_rings_I),
            collocation_I=transform_points(ref.collocation_I),
            collocation_velocity_I=point_velocity(ref.collocation_I),
            normals_I=transform_vectors(ref.normals_I),
            areas=ref.areas,  # rigid: areas unchanged
            leading_edge_I=transform_points(ref.leading_edge_I),
            trailing_edge_I=transform_points(ref.trailing_edge_I),
            leading_velocity_I=point_velocity(ref.leading_edge_I),
            trailing_velocity_I=point_velocity(ref.trailing_edge_I),
            chordwise_panels=ref.chordwise_panels,
            spanwise_panels=ref.spanwise_panels,
            topology_digest=ref.topology_digest,
        )

    def _coerce(self, value: Any) -> torch.Tensor:
        """Motion-law outputs may be lists/arrays; align with the reference."""
        return torch.as_tensor(
            value,
            dtype=self._ref.panel_rings_I.dtype,
            device=self._ref.panel_rings_I.device,
        )

    @staticmethod
    def _quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
        w, x, y, z = q[0], q[1], q[2], q[3]
        return torch.tensor(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ],
            device=q.device,
            dtype=q.dtype,
        )
