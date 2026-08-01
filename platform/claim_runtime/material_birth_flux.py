"""Conservation-derived material birth flux for a sharp wake junction.

This module implements only the local S3ae representation identity

    g_body - g_released = sign * dot(Gamma_g) * dt

with the forming-sheet state supplied by the validated finite-angle
sharp-edge oracle.  It does not modify the actual-boundary equation, infer a
finite-base state, evaluate velocity/pressure/force, or regularize a
singularity.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .distributed_doublet import (
    MaterialWakeBand,
    newborn_material_wake_band,
)
from .finite_angle_sheet_formation import (
    FiniteAngleSheetError,
    FiniteAngleSheetFormation,
    finite_angle_sheet_formation,
)


@dataclass(frozen=True)
class MaterialBirthFlux:
    """One nondegenerate finite-angle material birth state."""

    formation: FiniteAngleSheetFormation
    timestep: float
    repository_birth_flux_sign: int
    birth_flux: float
    newborn_length: float
    released_trace: np.ndarray
    midpoint_trace: np.ndarray
    current_trace: np.ndarray
    band: MaterialWakeBand


def consistent_p2_line_mass(span_vertices: Any) -> np.ndarray:
    """Return the assembled consistent mass matrix for a P2 line trace."""
    vertices = np.asarray(span_vertices, dtype=float)
    if (
        vertices.ndim != 1
        or len(vertices) < 2
        or not np.all(np.isfinite(vertices))
        or np.any(np.diff(vertices) <= 0.0)
    ):
        raise FiniteAngleSheetError(
            "span_vertices must be finite and strictly increasing"
        )
    count = 2 * len(vertices) - 1
    mass = np.zeros((count, count), dtype=float)
    reference = np.array(
        (
            (4.0, 2.0, -1.0),
            (2.0, 16.0, 2.0),
            (-1.0, 2.0, 4.0),
        ),
        dtype=float,
    )
    for interval, length in enumerate(np.diff(vertices)):
        dofs = np.array(
            (2 * interval, 2 * interval + 1, 2 * interval + 2),
            dtype=np.int64,
        )
        mass[np.ix_(dofs, dofs)] += float(length) * reference / 30.0
    return mass


def finite_angle_material_birth_flux(
    *,
    u1_plus: float,
    u2_minus: float,
    wedge_angle_deg: float,
    timestep: float,
    span_vertices: Any,
    repository_birth_flux_sign: int,
    released_trace: Any | None = None,
    sheet_id: str = "finite-angle-material-birth",
) -> MaterialBirthFlux:
    """Build the midpoint-P2 newborn strip from material circulation flux."""
    dt = float(timestep)
    if not np.isfinite(dt) or dt <= 0.0:
        raise FiniteAngleSheetError(
            "timestep must be finite and strictly positive"
        )
    sign = repository_birth_flux_sign
    if (
        not isinstance(sign, (int, np.integer))
        or int(sign) not in (-1, 1)
    ):
        raise FiniteAngleSheetError(
            "repository_birth_flux_sign must be exactly -1 or +1"
        )
    span = np.asarray(span_vertices, dtype=float)
    consistent_p2_line_mass(span)
    formation = finite_angle_sheet_formation(
        u1_plus=u1_plus,
        u2_minus=u2_minus,
        wedge_angle_deg=wedge_angle_deg,
    )
    if (
        not formation.state_identifiable
        or formation.relative_velocity is None
        or formation.relative_velocity <= 0.0
    ):
        raise FiniteAngleSheetError(
            "material birth geometry needs an identifiable positive "
            "finite-angle relative velocity"
        )
    cut_count = 2 * len(span) - 1
    if released_trace is None:
        released = np.zeros(cut_count, dtype=float)
    else:
        released = np.asarray(released_trace, dtype=float)
        if (
            released.shape != (cut_count,)
            or not np.all(np.isfinite(released))
        ):
            raise FiniteAngleSheetError(
                "released_trace must match the spanwise P2 trace"
            )
        released = released.copy()
    birth_flux = float(
        int(sign) * formation.circulation_rate
    )
    midpoint = released + 0.5 * dt * birth_flux
    current = released + dt * birth_flux
    length = float(formation.relative_velocity * dt)
    previous_edge = np.column_stack(
        (
            np.full(len(span), length),
            span,
            np.zeros(len(span)),
        )
    )
    current_edge = np.column_stack(
        (
            np.zeros(len(span)),
            span,
            np.zeros(len(span)),
        )
    )
    band = newborn_material_wake_band(
        sheet_id=sheet_id,
        vortex_family="TEV",
        previous_edge=previous_edge,
        current_edge=current_edge,
        time_nodes=np.array((0.0, 0.5 * dt, dt)),
        potential_jump_rows=np.vstack(
            (released, midpoint, current)
        ),
        span_diagonal_pattern="forward",
    )
    return MaterialBirthFlux(
        formation=formation,
        timestep=dt,
        repository_birth_flux_sign=int(sign),
        birth_flux=birth_flux,
        newborn_length=length,
        released_trace=released.copy(),
        midpoint_trace=midpoint.copy(),
        current_trace=current.copy(),
        band=band,
    )
