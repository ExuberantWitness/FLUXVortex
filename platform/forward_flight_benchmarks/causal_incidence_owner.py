"""Causal attached/transient/persistent incidence ownership for FluxV v4.

The v3 benchmark used a full-cycle mean and was therefore non-causal.  Here a
two-pole state observes only current and past local incidence.  The poles are
the slow circulation-state poles ``-0.30`` and ``-0.045`` published by
Izraelevitz et al. (2017), so no benchmark-force fit defines the memory.

The persistent fraction is the magnitude of the filtered signed incidence
divided by the same filter applied to absolute incidence.  A zero-mean fast
oscillation tends toward the transient owner, while a sustained installed
incidence tends toward the persistent separated-polar owner.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class CausalIncidenceOwnerParameters:
    fast_pole_per_convective_time: float = 0.30
    slow_pole_per_convective_time: float = 0.045
    incidence_floor_rad: float = 1.0e-10

    def __post_init__(self) -> None:
        if self.fast_pole_per_convective_time <= 0.0:
            raise ValueError("fast pole must be positive")
        if self.slow_pole_per_convective_time <= 0.0:
            raise ValueError("slow pole must be positive")
        if self.incidence_floor_rad < 0.0:
            raise ValueError("incidence floor must be nonnegative")


DEFAULT_CAUSAL_OWNER = CausalIncidenceOwnerParameters()


def causal_incidence_persistence(
    alpha_rad: np.ndarray,
    *,
    delta_time_convective: float,
    strip_weights: np.ndarray | None = None,
    parameters: CausalIncidenceOwnerParameters = DEFAULT_CAUSAL_OWNER,
) -> dict[str, np.ndarray | float | dict[str, float]]:
    """Run the two-pole incidence owner on a complete causal history.

    ``alpha_rad`` has shape ``(time, strip)``.  The state starts at zero; a
    caller evaluating an established periodic experiment must prepend the
    actual or repeated warm-up history rather than using future samples.
    """

    alpha = np.asarray(alpha_rad, dtype=float)
    if alpha.ndim == 1:
        alpha = alpha[:, None]
    if alpha.ndim != 2 or alpha.shape[0] < 2 or alpha.shape[1] < 1:
        raise ValueError("alpha_rad must have shape (time>=2, strip>=1)")
    if np.any(~np.isfinite(alpha)):
        raise FloatingPointError("incidence history contains non-finite values")
    if not np.isfinite(delta_time_convective) or delta_time_convective <= 0.0:
        raise ValueError("convective time step must be finite and positive")
    if strip_weights is None:
        weights = np.ones(alpha.shape[1], dtype=float)
    else:
        weights = np.asarray(strip_weights, dtype=float)
        if weights.shape != (alpha.shape[1],):
            raise ValueError("strip weights must have shape (strip,)")
        if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("strip weights must be finite and nonnegative")
        if not np.any(weights > 0.0):
            raise ValueError("at least one strip weight must be positive")

    fast_gain = 1.0 - np.exp(
        -parameters.fast_pole_per_convective_time * delta_time_convective
    )
    slow_gain = 1.0 - np.exp(
        -parameters.slow_pole_per_convective_time * delta_time_convective
    )
    signed_fast = np.zeros(alpha.shape[1])
    signed_slow = np.zeros(alpha.shape[1])
    magnitude_fast = np.zeros(alpha.shape[1])
    magnitude_slow = np.zeros(alpha.shape[1])
    strip_persistence = np.zeros_like(alpha)
    global_persistence = np.zeros(alpha.shape[0])

    for time_index, current_alpha in enumerate(alpha):
        signed_fast += fast_gain * (current_alpha - signed_fast)
        signed_slow += slow_gain * (signed_fast - signed_slow)
        magnitude_fast += fast_gain * (np.abs(current_alpha) - magnitude_fast)
        magnitude_slow += slow_gain * (magnitude_fast - magnitude_slow)
        active = magnitude_slow > parameters.incidence_floor_rad
        strip_persistence[time_index, active] = np.clip(
            np.abs(signed_slow[active]) / magnitude_slow[active], 0.0, 1.0
        )
        global_persistence[time_index] = np.average(
            strip_persistence[time_index], weights=weights
        )

    return {
        "global_persistence": global_persistence,
        "strip_persistence": strip_persistence,
        "filtered_signed_incidence_rad": signed_slow.copy(),
        "filtered_absolute_incidence_rad": magnitude_slow.copy(),
        "parameters": asdict(parameters),
        "delta_time_convective": float(delta_time_convective),
        "semantics": (
            "causal two-pole signed/absolute incidence owner; poles taken "
            "from Izraelevitz 2017 circulation-state dynamics"
        ),
    }
