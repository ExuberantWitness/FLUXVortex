"""Source-time causal incidence owner for an isolated FluxV v5d shadow.

This module is the source-time replacement candidate for the older diagnostic
in :mod:`causal_incidence_owner`.  Its only input clock is the local,
semi-chord convective increment

``Delta t_tilde = 2 |V_rel,0.75c,perp span| dt / c_local``.

The increment is supplied independently at every time and strip.  Two
source-frozen stable poles update identical signed-incidence and
absolute-incidence cascades.  Their slow-state ratio is an incidence-sign
coherence diagnostic; it is not a force model, a separation criterion, or a
published LEV state.

No paper name, case identity, force observation, phase wrap, or future sample
enters the update.  A periodic caller must prepend a real or repeated warm-up
history if it wants an established periodic state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


SOURCE_FAST_POLE_PER_T_TILDE = 0.30
SOURCE_SLOW_POLE_PER_T_TILDE = 0.045


@dataclass(frozen=True)
class SourceTimeOwnerParameters:
    """Source-frozen poles and an auditable unit/provenance manifest."""

    fast_pole_per_t_tilde: float = SOURCE_FAST_POLE_PER_T_TILDE
    slow_pole_per_t_tilde: float = SOURCE_SLOW_POLE_PER_T_TILDE

    def __post_init__(self) -> None:
        for value, name in (
            (self.fast_pole_per_t_tilde, "fast pole"),
            (self.slow_pole_per_t_tilde, "slow pole"),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            self.fast_pole_per_t_tilde != SOURCE_FAST_POLE_PER_T_TILDE
            or self.slow_pole_per_t_tilde != SOURCE_SLOW_POLE_PER_T_TILDE
        ):
            raise ValueError(
                "v5d source-time owner poles are source-frozen at 0.30 and "
                "0.045 per t_tilde; custom poles require another model identity"
            )

    def manifest(self) -> dict[str, float | str]:
        """Return the frozen parameter, source, clock, and scope contract."""

        return {
            "fast_pole_per_t_tilde": self.fast_pole_per_t_tilde,
            "slow_pole_per_t_tilde": self.slow_pole_per_t_tilde,
            "pole_units": "1/t_tilde (dimensionless inverse semi-chord time)",
            "pole_source": (
                "Izraelevitz, Zhu, and Triantafyllou (2017), Table 1; "
                "positive decay magnitudes 0.30 and 0.045"
            ),
            "delta_t_tilde_definition": (
                "2*|V_rel at 0.75c, perpendicular to local span|*dt/c_local"
            ),
            "delta_t_tilde_topology": "time-by-strip; one local value per update",
            "update_formula": (
                "x_fast[n]=exp(-0.30*Delta t_tilde[n])*x_fast[n-1] + "
                "(1-exp(-0.30*Delta t_tilde[n]))*input[n]; "
                "x_slow uses pole 0.045 and the updated x_fast[n]"
            ),
            "owner_formula": "p=|signed_slow|/magnitude_slow; p=0 at zero magnitude",
            "initial_state": "zero; no future or periodic phase-wrap initialization",
            "disabled_policy": (
                "return exact zero persistence; inspect alpha topology only; "
                "do not evaluate alpha values, Delta t_tilde, or strip weights"
            ),
            "force_scope": "diagnostic owner only; computes no aerodynamic force",
            "observation_access": "none",
            "observation_fit": "none",
            "case_or_paper_branch": "none",
        }


DEFAULT_SOURCE_TIME_OWNER_PARAMETERS = SourceTimeOwnerParameters()


def _history_topology(value: Any) -> tuple[int, int]:
    shape = np.shape(value)
    if len(shape) != 2 or shape[0] < 1 or shape[1] < 1:
        raise ValueError("alpha_rad must have shape (time>=1, strip>=1)")
    return int(shape[0]), int(shape[1])


def _disabled_result(
    topology: tuple[int, int],
    parameters: SourceTimeOwnerParameters,
) -> dict[str, Any]:
    time_count, strip_count = topology
    strip_zeros = np.zeros((time_count, strip_count), dtype=float)
    global_zeros = np.zeros(time_count, dtype=float)
    return {
        "global_persistence": global_zeros,
        "strip_persistence": strip_zeros.copy(),
        "signed_fast_history_rad": strip_zeros.copy(),
        "signed_slow_history_rad": strip_zeros.copy(),
        "magnitude_fast_history_rad": strip_zeros.copy(),
        "magnitude_slow_history_rad": strip_zeros.copy(),
        "normalized_strip_weights": None,
        "final_state": {
            "signed_fast_rad": np.zeros(strip_count, dtype=float),
            "signed_slow_rad": np.zeros(strip_count, dtype=float),
            "magnitude_fast_rad": np.zeros(strip_count, dtype=float),
            "magnitude_slow_rad": np.zeros(strip_count, dtype=float),
        },
        "parameters": parameters.manifest(),
        "diagnostics": {
            "enabled": False,
            "status": "not_evaluated_disabled",
            "state_updated": False,
            "causal": True,
            "lookahead_samples": 0,
        },
        "model_contract": {
            "shadow_only": True,
            "canonical_eligible": False,
            "claim_scope": "source-time incidence-owner mechanics only",
        },
    }


def _normalized_weights(value: Any, strip_count: int) -> np.ndarray:
    if value is None:
        return np.full(strip_count, 1.0 / strip_count, dtype=float)
    weights = np.asarray(value, dtype=float)
    if weights.shape != (strip_count,):
        raise ValueError("strip_weights must have shape (strip,)")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("strip_weights must be finite and nonnegative")
    maximum = float(np.max(weights))
    if maximum <= 0.0:
        raise ValueError("at least one strip weight must be positive")
    # Scaling before summation avoids overflow for otherwise valid large
    # relative weights and does not change the weighted average.
    scaled = weights / maximum
    normalized = scaled / np.sum(scaled)
    if np.any(~np.isfinite(normalized)):
        raise ValueError("normalized strip weights are not finite")
    return normalized


def source_time_causal_persistence(
    alpha_rad: Any,
    *,
    delta_t_tilde: Any,
    strip_weights: Any = None,
    enabled: bool = True,
    parameters: SourceTimeOwnerParameters = DEFAULT_SOURCE_TIME_OWNER_PARAMETERS,
) -> dict[str, Any]:
    """Run the source-clock incidence owner on a time-by-strip history.

    ``delta_t_tilde[n, s]`` is the positive elapsed semi-chord convective time
    used to assimilate ``alpha_rad[n, s]``.  The two cascades begin at zero.
    Each output row therefore depends only on rows ``0..n`` of the inputs.

    When disabled, the function deliberately reads only ``alpha_rad.shape``
    to size exact zero outputs.  It does not convert or inspect alpha values,
    ``delta_t_tilde``, or ``strip_weights``.
    """

    if not isinstance(enabled, (bool, np.bool_)):
        raise ValueError("enabled must be Boolean")
    topology = _history_topology(alpha_rad)
    if not enabled:
        return _disabled_result(topology, parameters)

    alpha = np.asarray(alpha_rad, dtype=float)
    convective_step = np.asarray(delta_t_tilde, dtype=float)
    if convective_step.shape != topology:
        raise ValueError("delta_t_tilde must match alpha_rad time-by-strip topology")
    if np.any(~np.isfinite(alpha)):
        raise ValueError("alpha_rad must be finite")
    if np.any(~np.isfinite(convective_step)) or np.any(convective_step <= 0.0):
        raise ValueError("delta_t_tilde must be finite and positive")

    time_count, strip_count = topology
    normalized_weights = _normalized_weights(strip_weights, strip_count)
    signed_fast = np.zeros(strip_count, dtype=float)
    signed_slow = np.zeros(strip_count, dtype=float)
    magnitude_fast = np.zeros(strip_count, dtype=float)
    magnitude_slow = np.zeros(strip_count, dtype=float)
    signed_fast_history = np.zeros_like(alpha)
    signed_slow_history = np.zeros_like(alpha)
    magnitude_fast_history = np.zeros_like(alpha)
    magnitude_slow_history = np.zeros_like(alpha)
    strip_persistence = np.zeros_like(alpha)

    for time_index in range(time_count):
        step = convective_step[time_index]
        # -expm1(-x) retains the small positive gain when the local convective
        # step is much smaller than one semi-chord time.
        fast_gain = -np.expm1(-parameters.fast_pole_per_t_tilde * step)
        slow_gain = -np.expm1(-parameters.slow_pole_per_t_tilde * step)
        current_alpha = alpha[time_index]
        with np.errstate(over="ignore", invalid="ignore"):
            signed_fast += fast_gain * (current_alpha - signed_fast)
            magnitude_fast += fast_gain * (np.abs(current_alpha) - magnitude_fast)
            signed_slow += slow_gain * (signed_fast - signed_slow)
            magnitude_slow += slow_gain * (magnitude_fast - magnitude_slow)

        for value, name in (
            (signed_fast, "signed fast state"),
            (signed_slow, "signed slow state"),
            (magnitude_fast, "magnitude fast state"),
            (magnitude_slow, "magnitude slow state"),
        ):
            if np.any(~np.isfinite(value)):
                raise ValueError(f"{name} is not finite")

        active = magnitude_slow > 0.0
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            active_ratio = np.abs(signed_slow[active]) / magnitude_slow[active]
        if np.any(~np.isfinite(active_ratio)):
            raise ValueError("strip persistence ratio is not finite")
        strip_persistence[time_index, active] = np.minimum(active_ratio, 1.0)
        signed_fast_history[time_index] = signed_fast
        signed_slow_history[time_index] = signed_slow
        magnitude_fast_history[time_index] = magnitude_fast
        magnitude_slow_history[time_index] = magnitude_slow

    global_persistence = strip_persistence @ normalized_weights
    if np.any(~np.isfinite(global_persistence)):
        raise ValueError("global persistence is not finite")

    return {
        "global_persistence": global_persistence,
        "strip_persistence": strip_persistence,
        "signed_fast_history_rad": signed_fast_history,
        "signed_slow_history_rad": signed_slow_history,
        "magnitude_fast_history_rad": magnitude_fast_history,
        "magnitude_slow_history_rad": magnitude_slow_history,
        "normalized_strip_weights": normalized_weights,
        "final_state": {
            "signed_fast_rad": signed_fast.copy(),
            "signed_slow_rad": signed_slow.copy(),
            "magnitude_fast_rad": magnitude_fast.copy(),
            "magnitude_slow_rad": magnitude_slow.copy(),
        },
        "parameters": parameters.manifest(),
        "diagnostics": {
            "enabled": True,
            "status": "source_time_causal_owner_evaluated",
            "state_updated": True,
            "causal": True,
            "lookahead_samples": 0,
        },
        "model_contract": {
            "shadow_only": True,
            "canonical_eligible": False,
            "claim_scope": "source-time incidence-owner mechanics only",
        },
    }


__all__ = [
    "DEFAULT_SOURCE_TIME_OWNER_PARAMETERS",
    "SOURCE_FAST_POLE_PER_T_TILDE",
    "SOURCE_SLOW_POLE_PER_T_TILDE",
    "SourceTimeOwnerParameters",
    "source_time_causal_persistence",
]
