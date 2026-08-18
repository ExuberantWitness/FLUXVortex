"""Observation-free A/L/P region-owner shadow for FluxV v5d2.

This module transfers only the *cross-section regions* implied by Yang et al.
(2025), Eqs. (11)--(12), to an area-weighted spanwise ownership diagnostic.
For every time and strip, with ``C_alpha`` frozen at the published value five,

``A: |alpha| <= alpha_sep``,
``L: alpha_sep < |alpha| <= C_alpha * alpha_sep``, and
``P: |alpha| > C_alpha * alpha_sep``.

The transfer from a two-dimensional sectional PLEV law to three-dimensional
owner fractions is an explicitly non-canonical shadow.  It has no memory, no
case or paper branch, no observation access, and computes no aerodynamic
force.  A later force-owner proposal must be validated separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


YANG2025_C_ALPHA = 5.0


@dataclass(frozen=True)
class RegionOwnerParameters:
    """Frozen source parameter for the v5d2 region-owner shadow."""

    c_alpha: float = YANG2025_C_ALPHA

    def __post_init__(self) -> None:
        if not np.isfinite(self.c_alpha) or self.c_alpha <= 0.0:
            raise ValueError("C_alpha must be finite and positive")
        if self.c_alpha != YANG2025_C_ALPHA:
            raise ValueError(
                "v5d2 freezes C_alpha=5 from Yang et al. (2025), "
                "Eqs. (11)--(12); a custom value requires another model identity"
            )

    def manifest(self) -> dict[str, float | str]:
        """Return the source, transfer, and non-force scope declaration."""

        return {
            "c_alpha": self.c_alpha,
            "c_alpha_source": "Yang et al. (2025), Eqs. (11)--(12)",
            "region_A": "|alpha| <= alpha_sep",
            "region_L": "alpha_sep < |alpha| <= C_alpha*alpha_sep",
            "region_P": "|alpha| > C_alpha*alpha_sep",
            "aggregation": "strip-area-weighted region fractions",
            "transfer_status": "cross-section region transfer shadow",
            "force_scope": "none; computes no aerodynamic force",
            "observation_access": "none",
            "observation_fit": "none",
            "case_or_paper_branch": "none",
        }


DEFAULT_REGION_OWNER_PARAMETERS = RegionOwnerParameters()


def _history_topology(value: Any) -> tuple[int, int]:
    shape = np.shape(value)
    if len(shape) != 2 or shape[0] < 1 or shape[1] < 1:
        raise ValueError("alpha_rad must have shape (time>=1, strip>=1)")
    return int(shape[0]), int(shape[1])


def _disabled_result(
    topology: tuple[int, int], parameters: RegionOwnerParameters
) -> dict[str, Any]:
    time_count, strip_count = topology
    ones = np.ones(time_count, dtype=float)
    zeros = np.zeros(time_count, dtype=float)
    strip_ones = np.ones((time_count, strip_count), dtype=bool)
    strip_zeros = np.zeros((time_count, strip_count), dtype=bool)
    return {
        "weights": {"wA": ones, "wL": zeros.copy(), "wP": zeros.copy()},
        "strip_region_masks": {
            "A": strip_ones,
            "L": strip_zeros.copy(),
            "P": strip_zeros.copy(),
        },
        "normalized_strip_weights": None,
        "alpha_sep_rad_by_strip": None,
        "parameters": parameters.manifest(),
        "diagnostics": {
            "enabled": False,
            "status": "not_evaluated_disabled",
            "causal": True,
            "lookahead_samples": 0,
            "max_abs_weight_sum_residual": 0.0,
        },
        "model_contract": {
            "shadow_only": True,
            "transfer_scope": "cross_section_regions_to_spanwise_area_fractions",
            "canonical_eligible": False,
            "computes_force": False,
            "observation_access": "none",
            "claim_scope": "region-owner mechanics only",
        },
    }


def _separation_by_strip(value: Any, strip_count: int) -> np.ndarray:
    separation = np.asarray(value, dtype=float)
    if separation.ndim == 0:
        separation = np.full(strip_count, float(separation), dtype=float)
    elif separation.shape != (strip_count,):
        raise ValueError("alpha_sep_rad must be a scalar or have shape (strip,)")
    if np.any(~np.isfinite(separation)) or np.any(separation <= 0.0):
        raise ValueError("alpha_sep_rad must be finite and positive")
    return separation


def _normalized_area_weights(value: Any, strip_count: int) -> np.ndarray:
    weights = np.asarray(value, dtype=float)
    if weights.shape != (strip_count,):
        raise ValueError("strip_weights must have shape (strip,)")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("strip_weights must be finite and nonnegative")
    maximum = float(np.max(weights))
    if maximum <= 0.0:
        raise ValueError("at least one strip weight must be positive")
    scaled = weights / maximum
    normalized = scaled / np.sum(scaled)
    if np.any(~np.isfinite(normalized)):
        raise ValueError("normalized strip weights are not finite")
    return normalized


def cross_section_region_owner(
    alpha_rad: Any,
    *,
    alpha_sep_rad: Any,
    strip_weights: Any,
    enabled: bool = True,
    parameters: RegionOwnerParameters = DEFAULT_REGION_OWNER_PARAMETERS,
) -> dict[str, Any]:
    """Return instantaneous, area-weighted A/L/P region-owner fractions.

    ``alpha_rad`` is a time-by-strip history.  ``alpha_sep_rad`` is either one
    positive scalar shared by all strips or one positive value per strip.
    ``strip_weights`` are relative strip areas; their absolute scale is
    irrelevant.

    Disabled operation reads only ``alpha_rad.shape`` so it can return an exact
    ``A=1, L=P=0`` identity.  It deliberately does not convert or inspect the
    alpha values, separation input, or strip weights.
    """

    if not isinstance(enabled, (bool, np.bool_)):
        raise ValueError("enabled must be Boolean")
    topology = _history_topology(alpha_rad)
    if not enabled:
        return _disabled_result(topology, parameters)

    alpha = np.asarray(alpha_rad, dtype=float)
    if np.any(~np.isfinite(alpha)):
        raise ValueError("alpha_rad must be finite")

    _, strip_count = topology
    separation = _separation_by_strip(alpha_sep_rad, strip_count)
    normalized_weights = _normalized_area_weights(strip_weights, strip_count)
    with np.errstate(over="ignore", invalid="ignore"):
        separated_threshold = parameters.c_alpha * separation
    if np.any(~np.isfinite(separated_threshold)):
        raise ValueError("C_alpha*alpha_sep_rad must be finite")

    magnitude = np.abs(alpha)
    region_a = magnitude <= separation[None, :]
    region_l = (magnitude > separation[None, :]) & (
        magnitude <= separated_threshold[None, :]
    )
    region_p = magnitude > separated_threshold[None, :]
    partition_count = (
        region_a.astype(np.int8) + region_l.astype(np.int8) + region_p.astype(np.int8)
    )
    if np.any(partition_count != 1):
        raise ValueError("A/L/P strip regions do not form an exclusive partition")

    fractions = np.column_stack(
        (
            region_a @ normalized_weights,
            region_l @ normalized_weights,
            region_p @ normalized_weights,
        )
    )
    fraction_sums = np.sum(fractions, axis=1)
    if (
        np.any(~np.isfinite(fractions))
        or np.any(fractions < 0.0)
        or np.any(fraction_sums <= 0.0)
    ):
        raise ValueError("area-weighted owner fractions are invalid")
    # This normalization only closes floating-point summation of an already
    # exclusive partition; it introduces no threshold or tunable parameter.
    fractions /= fraction_sums[:, None]
    weight_sum_residual = np.sum(fractions, axis=1) - 1.0

    return {
        "weights": {
            "wA": fractions[:, 0],
            "wL": fractions[:, 1],
            "wP": fractions[:, 2],
        },
        "strip_region_masks": {"A": region_a, "L": region_l, "P": region_p},
        "normalized_strip_weights": normalized_weights,
        "alpha_sep_rad_by_strip": separation,
        "parameters": parameters.manifest(),
        "diagnostics": {
            "enabled": True,
            "status": "cross_section_transfer_shadow_evaluated",
            "causal": True,
            "lookahead_samples": 0,
            "max_abs_weight_sum_residual": float(np.max(np.abs(weight_sum_residual))),
        },
        "model_contract": {
            "shadow_only": True,
            "transfer_scope": "cross_section_regions_to_spanwise_area_fractions",
            "canonical_eligible": False,
            "computes_force": False,
            "observation_access": "none",
            "claim_scope": "region-owner mechanics only",
        },
    }


__all__ = [
    "DEFAULT_REGION_OWNER_PARAMETERS",
    "YANG2025_C_ALPHA",
    "RegionOwnerParameters",
    "cross_section_region_owner",
]
