"""Finite-angle sharp-edge vortex-sheet formation identities.

This is the S2c algebraic oracle for claim ``N3.1j3b6d``.  It implements
the high-Reynolds-number finite-angle trailing-edge relations derived by
Xia & Mohseni (JFM 830, 2017): angle closure, unsteady Kutta strength,
circulation rate, and mass/momentum-compatible sheet-side velocities.

It does not represent a blunt base, viscous sheet inventory, 3-D junction,
body boundary solve, wake time integration, or production load.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class FiniteAngleSheetError(ValueError):
    """Invalid state for the finite-angle sharp-edge canonical."""


@dataclass(frozen=True)
class FiniteAngleSheetFormation:
    u1_plus: float
    u2_minus: float
    wedge_angle_rad: float
    delta_theta1: float
    delta_theta2: float
    sheet_strength: float
    circulation_rate: float
    relative_velocity: float | None
    u_g_plus: float | None
    u_g_minus: float | None
    state_identifiable: bool
    angle_sum_residual: float
    direction_residual: float
    kutta_strength_residual: float
    circulation_rate_residual: float | None
    momentum_residual: float | None
    normalized_angle_sum_residual: float
    normalized_direction_residual: float
    normalized_kutta_strength_residual: float
    normalized_circulation_rate_residual: float | None
    normalized_momentum_residual: float | None


def finite_angle_sheet_formation(
    *,
    u1_plus: float,
    u2_minus: float,
    wedge_angle_deg: float,
) -> FiniteAngleSheetFormation:
    """Solve the parameter-free finite-angle formation identities."""

    first = float(u1_plus)
    second = float(u2_minus)
    wedge = np.deg2rad(float(wedge_angle_deg))
    if not np.isfinite(first) or not np.isfinite(second):
        raise FiniteAngleSheetError(
            "u1_plus and u2_minus must be finite"
        )
    if first > 0.0 or second > 0.0:
        raise FiniteAngleSheetError(
            "the canonical shedding convention requires both velocities <= 0"
        )
    if first == 0.0 and second == 0.0:
        raise FiniteAngleSheetError(
            "at least one incident-side velocity must be nonzero"
        )
    if not np.isfinite(wedge) or not (0.0 < wedge < np.pi):
        raise FiniteAngleSheetError(
            "wedge_angle_deg must map into (0,180) degrees"
        )

    first_speed = -first
    second_speed = -second
    theta1 = float(np.arctan2(
        second_speed * np.sin(wedge),
        first_speed + second_speed * np.cos(wedge),
    ))
    theta2 = float(wedge - theta1)
    strength_identity = (
        first * np.cos(theta1) - second * np.cos(theta2)
    )
    circulation_rate = 0.5 * (second**2 - first**2)

    speed_scale = max(first_speed + second_speed, np.finfo(float).tiny)
    strength_scale = max(
        abs(first * np.cos(theta1))
        + abs(second * np.cos(theta2)),
        speed_scale,
    )
    rate_scale = max(first**2 + second**2, np.finfo(float).tiny)
    momentum_scale = max(speed_scale**3, np.finfo(float).tiny)
    degeneracy_floor = (
        64.0 * np.finfo(float).eps * strength_scale
    )
    identifiable = not (
        abs(strength_identity) <= degeneracy_floor
        and abs(circulation_rate)
        <= 64.0 * np.finfo(float).eps * rate_scale
    )

    relative_velocity: float | None
    u_g_plus: float | None
    u_g_minus: float | None
    circulation_residual: float | None
    momentum_residual: float | None
    normalized_circulation: float | None
    normalized_momentum: float | None
    if identifiable:
        relative_velocity = float(
            circulation_rate / strength_identity
        )
        u_g_plus = float(relative_velocity + 0.5 * strength_identity)
        u_g_minus = float(relative_velocity - 0.5 * strength_identity)
        circulation_residual = float(
            relative_velocity * strength_identity - circulation_rate
        )
        momentum_residual = float(
            first * second * np.sin(wedge)
            + first * u_g_plus * np.sin(theta1)
            + second * u_g_minus * np.sin(theta2)
        )
        normalized_circulation = float(
            abs(circulation_residual) / rate_scale
        )
        normalized_momentum = float(
            abs(momentum_residual) / momentum_scale
        )
    else:
        relative_velocity = None
        u_g_plus = None
        u_g_minus = None
        circulation_residual = None
        momentum_residual = None
        normalized_circulation = None
        normalized_momentum = None

    angle_residual = float(theta1 + theta2 - wedge)
    direction_residual = float(
        first * np.sin(theta1) - second * np.sin(theta2)
    )
    strength = float(strength_identity)
    kutta_residual = float(
        strength
        - (
            first * np.cos(theta1)
            - second * np.cos(theta2)
        )
    )
    return FiniteAngleSheetFormation(
        u1_plus=first,
        u2_minus=second,
        wedge_angle_rad=float(wedge),
        delta_theta1=theta1,
        delta_theta2=theta2,
        sheet_strength=strength,
        circulation_rate=float(circulation_rate),
        relative_velocity=relative_velocity,
        u_g_plus=u_g_plus,
        u_g_minus=u_g_minus,
        state_identifiable=identifiable,
        angle_sum_residual=angle_residual,
        direction_residual=direction_residual,
        kutta_strength_residual=kutta_residual,
        circulation_rate_residual=circulation_residual,
        momentum_residual=momentum_residual,
        normalized_angle_sum_residual=float(
            abs(angle_residual) / max(abs(wedge), 1.0)
        ),
        normalized_direction_residual=float(
            abs(direction_residual) / speed_scale
        ),
        normalized_kutta_strength_residual=float(
            abs(kutta_residual) / strength_scale
        ),
        normalized_circulation_rate_residual=normalized_circulation,
        normalized_momentum_residual=normalized_momentum,
    )
