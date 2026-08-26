"""Izraelevitz-Scherer 2017: 12 motion conditions x 14 markers, hovering flapper.

U3 (plan §11.4): the elliptical wing with heave/pitch and phase offset emits
``SurfaceFrame`` directly on native 3D V5M.  The post-hoc separation
correction must become live solver state, and the 14-marker ground truth
must not be double-counted as independent conditions (12 unique motion
conditions only).  The case registry stays empty until the GT data adapter
is wired in a later U3 sub-slice.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IzraCaseConfig:
    case_id: str
    # 12 unique motion conditions, 14 marker measurements

    description: str = "hovering flapper, heave/pitch with phase offset"


IZRA_CASES = {}  # filled when the GT data adapter is wired
