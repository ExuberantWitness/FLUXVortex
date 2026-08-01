"""Local conservation witnesses for blunt-base state nonuniqueness.

The outer-surface speed and actual NACA-2406 corner geometry are held fixed.
Only an intentionally unobserved base-side incident speed is varied.  Each
member is solved with the already-qualified finite-angle conservation
identities.  The family is an identifiability audit, not a closure model.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .blunt_base_topology import (
    BluntBaseTopology,
    naca4_blunt_base_topology,
)
from .finite_angle_sheet_formation import (
    FiniteAngleSheetFormation,
    finite_angle_sheet_formation,
)


@dataclass(frozen=True)
class BluntBaseCornerWitness:
    base_speed_ratio: float
    observed_outer_speed: float
    upper_wedge_angle_deg: float
    lower_wedge_angle_deg: float
    upper: FiniteAngleSheetFormation
    lower: FiniteAngleSheetFormation


def _angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    return float(
        np.rad2deg(
            np.arccos(
                np.clip(float(np.dot(first, second)), -1.0, 1.0)
            )
        )
    )


def blunt_base_corner_witnesses(
    *,
    base_speed_ratios: tuple[float, ...],
    observed_outer_speed: float = 1.0,
    geometry: BluntBaseTopology | None = None,
) -> tuple[BluntBaseCornerWitness, ...]:
    """Return the fixed-outer-input, varying-base-state witness family."""

    outer = float(observed_outer_speed)
    if not np.isfinite(outer) or outer <= 0.0:
        raise ValueError("observed_outer_speed must be positive")
    if not base_speed_ratios:
        raise ValueError("base_speed_ratios must not be empty")
    ratios = tuple(float(value) for value in base_speed_ratios)
    if (
        not np.all(np.isfinite(ratios))
        or any(value < 0.0 or value >= 1.0 for value in ratios)
        or any(right <= left for left, right in zip(ratios, ratios[1:]))
    ):
        raise ValueError(
            "base_speed_ratios must be finite, ordered and in [0,1)"
        )
    topology = (
        naca4_blunt_base_topology(base_fraction=1.0)
        if geometry is None
        else geometry
    )
    if (
        not isinstance(topology, BluntBaseTopology)
        or topology.base_thickness <= 0.0
    ):
        raise ValueError("geometry must have a finite blunt base")
    base_down = (
        topology.lower_corner - topology.upper_corner
    ) / topology.base_thickness
    base_up = -base_down
    upper_wedge = _angle_deg(
        topology.upper_tangent, base_down
    )
    lower_wedge = _angle_deg(
        base_up, topology.lower_tangent
    )
    return tuple(
        BluntBaseCornerWitness(
            base_speed_ratio=ratio,
            observed_outer_speed=outer,
            upper_wedge_angle_deg=upper_wedge,
            lower_wedge_angle_deg=lower_wedge,
            upper=finite_angle_sheet_formation(
                u1_plus=-outer,
                u2_minus=-ratio * outer,
                wedge_angle_deg=upper_wedge,
            ),
            lower=finite_angle_sheet_formation(
                u1_plus=-outer,
                u2_minus=-ratio * outer,
                wedge_angle_deg=lower_wedge,
            ),
        )
        for ratio in ratios
    )
