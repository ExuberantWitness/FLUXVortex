"""Yang 2025 flapping wing: six installation angles, prescribed mechanism motion.

U3 (plan §11.3): the four-bar mechanism motion is emitted as
``PrescribedRigidSurfaceKinematics`` on native multi-surface V5M.  The six
installation angles from the paper's Fig. 11 must be reported per-angle
(lift/drag MAE each), never only as a cross-case aggregate.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class YangCaseConfig:
    case_id: str
    installation_angle_deg: float
    # Six angles from the paper's Fig. 11

    description: str = "four-bar mechanism flapping wing"


YANG_CASES = {
    f"YANG-{a}deg": YangCaseConfig(case_id=f"YANG-{a}deg", installation_angle_deg=a)
    for a in (0.0, 5.0, 10.0, 15.0, 20.0, 25.0)
}
