"""Multi-body surface kinematics: evaluates multiple SurfaceFrames from
a shared body pose + per-surface joint angles.

U4 (plan §7.1/§7.2): one aircraft body owns the 6-DOF state; each wing is a
surface hanging off that body through its own joint.  Here every surface has
its own ``PrescribedRigidSurfaceKinematics`` (later, a Q16 surface), and they
share one body pose — identity/fixed in U4, a free 6-DOF body in U5.  The
kinematic chain of plan §7.2 (body translation, body angular rate cross
product, joint rate, elastic rate) is supplied by the per-surface kinematics;
this class only aggregates them in a stable surface order.
"""
from __future__ import annotations

from typing import Any


class MultiSurfaceKinematics:
    """Evaluates all surface frames for a multi-surface configuration.

    Each surface has its own PrescribedRigidSurfaceKinematics (or later,
    Q16 surface), but they share a common body pose. In U4, the body is
    fixed (identity pose); U5 makes it a free 6-DOF body.
    """

    def __init__(self, surface_kinematics_list: list[Any]) -> None:
        if not surface_kinematics_list:
            raise ValueError("MultiSurfaceKinematics requires at least one surface")
        self._kinematics = list(surface_kinematics_list)

    @property
    def surface_ids(self) -> tuple[str, ...]:
        return tuple(k.surface_ids[0] for k in self._kinematics)

    def evaluate(self, body_pose, time: float) -> tuple[Any, ...]:
        """Evaluate all surfaces. body_pose is (pos, quat) — identity in U4.

        Per-surface kinematics may return a single ``SurfaceFrame`` (e.g.
        ``PrescribedRigidSurfaceKinematics``) or a tuple of frames; both are
        flattened in constructor order.
        """
        frames = []
        for kin in self._kinematics:
            result = kin.evaluate(time)
            if isinstance(result, (tuple, list)):
                frames.extend(result)
            else:
                frames.append(result)
        return tuple(frames)
