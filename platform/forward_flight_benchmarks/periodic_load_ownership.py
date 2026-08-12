"""Periodic attached/separated load ownership for the exploratory v3 model.

The full-angle polar in :mod:`uvlm_polar_correction` is an instantaneous
static-polar residual.  The v1 model assigned that residual's cycle mean to
all motions, including symmetric heave--pitch cycles with zero persistent
incidence.  The Izraelevitz Figure-14 experiment showed that this ownership
rule is not transferable.

This module implements a case-agnostic periodic ownership rule.  For every
strip, the fraction of incidence that persists through the cycle is

``abs(mean(alpha)) / mean(abs(alpha))``.

The strip fractions are area-weighted when weights are supplied.  Their
aggregate controls both the periodic mean owner and the alternating-load
gate: zero persistence selects the attached one-state ULLT history, while
unit persistence selects the separated UVLM/full-angle-polar history.  This
is an exploratory periodic two-pass rule developed after seeing Figure 14;
it is not an LEV, dynamic-stall, or causal transient closure.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def periodic_incidence_persistence(
    alpha_rad: np.ndarray,
    *,
    strip_weights: np.ndarray | None = None,
    incidence_floor_rad: float = 1.0e-12,
) -> tuple[float, np.ndarray]:
    """Return global and per-strip persistent-incidence fractions.

    Parameters
    ----------
    alpha_rad:
        Periodic local incidence with shape ``(phase, strip)``.
    strip_weights:
        Optional nonnegative strip weights, normally strip areas.  Equal
        weighting is used when omitted.
    incidence_floor_rad:
        Strips whose mean absolute incidence is below this numerical floor
        are assigned zero persistence.
    """

    alpha = np.asarray(alpha_rad, dtype=float)
    if alpha.ndim != 2 or alpha.shape[0] < 2 or alpha.shape[1] < 1:
        raise ValueError("alpha_rad must have shape (phase>=2, strip>=1)")
    if np.any(~np.isfinite(alpha)):
        raise ValueError("alpha_rad must be finite")
    if incidence_floor_rad < 0.0 or not np.isfinite(incidence_floor_rad):
        raise ValueError("incidence floor must be finite and nonnegative")

    strip_count = alpha.shape[1]
    if strip_weights is None:
        weights = np.ones(strip_count, dtype=float)
    else:
        weights = np.asarray(strip_weights, dtype=float)
        if weights.shape != (strip_count,):
            raise ValueError("strip_weights must have shape (strip,)")
        if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("strip_weights must be finite and nonnegative")
        if not np.any(weights > 0.0):
            raise ValueError("at least one strip weight must be positive")

    mean_signed_magnitude = np.abs(np.mean(alpha, axis=0))
    mean_absolute = np.mean(np.abs(alpha), axis=0)
    persistent = np.zeros(strip_count, dtype=float)
    active = (mean_absolute > incidence_floor_rad) & (
        mean_signed_magnitude > incidence_floor_rad
    )
    persistent[active] = mean_signed_magnitude[active] / mean_absolute[active]
    # Jensen's inequality bounds the ratio by one; clipping only removes
    # roundoff excursions and does not introduce a model threshold.
    persistent = np.clip(persistent, 0.0, 1.0)
    global_persistence = float(np.average(persistent, weights=weights))
    return global_persistence, persistent


def blend_periodic_persistent_owner(
    attached_history: dict[str, Any],
    separated_history: dict[str, Any],
    separation_fraction: np.ndarray,
    *,
    persistence_fraction: float,
    rho_kg_m3: float,
    freestream_m_s: float,
    area_m2: float,
) -> dict[str, Any]:
    """Blend ULLT and UVLM/polar with one periodic load owner.

    The cycle mean is ``(1-p) mean(ULLT) + p mean(UVLM+polar)``.  The usual
    attached/separated AC gate is attenuated to ``p*separation_fraction``.
    A zero-persistence cycle therefore reduces completely to ULLT instead of
    retaining a recentered static-polar drag history.  This is an exploratory
    two-pass construction, not a causal LEV/dynamic-stall closure.
    """

    phase = np.asarray(attached_history["phase"], dtype=float)
    other_phase = np.asarray(separated_history["phase"], dtype=float)
    if phase.ndim != 1 or phase.size < 2:
        raise ValueError("history phase must contain at least two samples")
    if phase.shape != other_phase.shape or not np.allclose(
        phase, other_phase, atol=1.0e-12, rtol=0.0
    ):
        raise ValueError("attached and separated phases are not aligned")
    separation = np.asarray(separation_fraction, dtype=float)
    if separation.shape != phase.shape or np.any(~np.isfinite(separation)):
        raise ValueError("separation_fraction must be a finite phase vector")
    if np.any((separation < 0.0) | (separation > 1.0)):
        raise ValueError("separation_fraction must lie in [0, 1]")
    persistence = float(persistence_fraction)
    if not np.isfinite(persistence) or not 0.0 <= persistence <= 1.0:
        raise ValueError("persistence_fraction must lie in [0, 1]")
    q_area = 0.5 * rho_kg_m3 * freestream_m_s**2 * area_m2
    if q_area <= 0.0 or not np.isfinite(q_area):
        raise ValueError("reference dynamic pressure times area must be positive")

    attached = np.column_stack(
        (
            np.asarray(attached_history["lift_n"], dtype=float),
            np.asarray(attached_history["drag_n"], dtype=float),
        )
    )
    separated = np.column_stack(
        (
            np.asarray(separated_history["lift_n"], dtype=float),
            np.asarray(separated_history["drag_n"], dtype=float),
        )
    )
    if attached.shape != (phase.size, 2) or separated.shape != attached.shape:
        raise ValueError("load histories must share the phase-vector length")
    if np.any(~np.isfinite(attached)) or np.any(~np.isfinite(separated)):
        raise ValueError("load histories must be finite")
    attached_mean = np.array(
        [
            float(attached_history.get("mean_lift_n", np.mean(attached[:, 0]))),
            float(attached_history.get("mean_drag_n", np.mean(attached[:, 1]))),
        ]
    )
    separated_mean = np.array(
        [
            float(separated_history.get("mean_lift_n", np.mean(separated[:, 0]))),
            float(separated_history.get("mean_drag_n", np.mean(separated[:, 1]))),
        ]
    )
    attached_ac = attached - np.mean(attached, axis=0)
    separated_ac = separated - np.mean(separated, axis=0)
    effective_separation = persistence * separation
    mixed_ac = (
        1.0 - effective_separation[:, None]
    ) * attached_ac + effective_separation[:, None] * separated_ac
    mixed_ac -= np.mean(mixed_ac, axis=0)
    target_mean = (1.0 - persistence) * attached_mean + persistence * separated_mean
    loads = target_mean + mixed_ac
    return {
        "phase": phase.copy(),
        "lift_n": loads[:, 0],
        "drag_n": loads[:, 1],
        "thrust_n": -loads[:, 1],
        "CL": loads[:, 0] / q_area,
        "CD": loads[:, 1] / q_area,
        "CT": -loads[:, 1] / q_area,
        "mean_lift_n": float(target_mean[0]),
        "mean_drag_n": float(target_mean[1]),
        "mean_thrust_n": float(-target_mean[1]),
        "mean_CL": float(target_mean[0] / q_area),
        "mean_CD": float(target_mean[1] / q_area),
        "mean_CT": float(-target_mean[1] / q_area),
        "persistence_fraction": persistence,
        "separation_fraction": separation.copy(),
        "effective_separation_fraction": effective_separation,
        "mean_owner": {
            "attached_ullt_weight": 1.0 - persistence,
            "separated_uvlm_polar_weight": persistence,
        },
        "model_semantics": (
            "exploratory periodic FluxV v3: persistent incidence blends the "
            "one-state ULLT and UVLM/polar mean owners and attenuates the "
            "separated AC gate; two-pass/post-hoc, no LEV-suction claim"
        ),
    }
