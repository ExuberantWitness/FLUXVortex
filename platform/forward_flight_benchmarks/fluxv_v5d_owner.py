"""Parameter-free exclusive force-owner ledger for the FluxV v5d0 shadow.

The shadow only rewrites the already-frozen v4b/v5c0 blend.  At every material
point it defines three complete candidate loads,

``A = UVLM``, ``L = A + delta_LDVM``, and ``P = A + delta_polar``,

and gives them mutually exclusive weights

``wP=p``, ``wL=(1-p)m``, and ``wA=(1-p)(1-m)``.

Here ``m`` is not a fitted gate or a persistent state.  It is the exact,
pointwise Boolean witness that at least one paired-LDVM component discrepancy
(``CNc``, ``CNnc``, ``CNnonl``, or ``CSf``) is nonzero.  Consequently this
module introduces no threshold, time scale, paper identifier, case branch, or
observation access.  It is an ownership/accounting shadow only; it is not a
new force model and is deliberately marked non-canonical.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


PAIRED_LDVM_COMPONENTS = ("CNc", "CNnc", "CNnonl", "CSf")


def _finite_array(name: str, value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _load_history(name: str, value: Any) -> np.ndarray:
    history = _finite_array(name, value)
    if history.ndim < 2 or history.shape[0] < 1 or history.shape[-1] < 1:
        raise ValueError(
            f"{name} must be a nonempty history with a final load-component axis"
        )
    return history


def _owner_field(name: str, value: Any, shape: tuple[int, ...]) -> np.ndarray:
    field = _finite_array(name, value)
    try:
        return np.broadcast_to(field, shape)
    except ValueError as error:
        raise ValueError(
            f"{name} cannot broadcast to owner topology {shape}"
        ) from error


def paired_ldvm_material_witness(
    component_discrepancy: Mapping[str, Any],
) -> np.ndarray:
    """Return the exact pointwise material witness for a paired-LDVM delta.

    The comparison is intentionally ``!= 0.0`` with no tolerance.  A tolerance
    would be an additional tunable ownership parameter and could suppress a
    real (albeit small) paired-model discrepancy.
    """

    try:
        components = tuple(
            _finite_array(
                f"paired-LDVM {name} discrepancy", component_discrepancy[name]
            )
            for name in PAIRED_LDVM_COMPONENTS
        )
    except (KeyError, TypeError) as error:
        required = ", ".join(PAIRED_LDVM_COMPONENTS)
        raise ValueError(
            f"component_discrepancy must contain paired-LDVM deltas: {required}"
        ) from error

    reference_shape = components[0].shape
    if not reference_shape or reference_shape[0] < 1:
        raise ValueError("paired-LDVM component histories must be nonempty")
    if any(component.shape != reference_shape for component in components[1:]):
        raise ValueError("paired-LDVM component histories must have identical shapes")

    return np.logical_or.reduce(tuple(component != 0.0 for component in components))


def _disabled_shadow(uvlm: np.ndarray) -> dict[str, Any]:
    owner_shape = uvlm.shape[:-1]
    zeros = np.zeros(owner_shape, dtype=float)
    ones = np.ones(owner_shape, dtype=float)
    zero_load = np.zeros_like(uvlm)
    return {
        "load": uvlm.copy(),
        "candidate_loads": {"A": uvlm.copy(), "L": uvlm.copy(), "P": uvlm.copy()},
        "weights": {"wA": ones, "wL": zeros.copy(), "wP": zeros.copy()},
        "material_witness": np.zeros(owner_shape, dtype=bool),
        "persistence": zeros.copy(),
        "weighted_owner_ledger": {
            "A": uvlm.copy(),
            "L": zero_load.copy(),
            "P": zero_load.copy(),
        },
        "residual_ledger": {
            "UVLM": uvlm.copy(),
            "paired_LDVM": zero_load.copy(),
            "polar": zero_load.copy(),
        },
        "diagnostics": {
            "enabled": False,
            "status": "not_evaluated_disabled",
            "max_abs_weight_sum_residual": 0.0,
            "max_abs_owner_residual_ledger_difference": 0.0,
            "max_abs_v4b_replay_difference": 0.0,
        },
        "model_contract": {
            "shadow_only": True,
            "evaluation_scope": "mechanical_shadow_only",
            "canonical_eligible": False,
            "parameters": {},
            "observation_access": "none",
            "claim_scope": "accounting_equivalence_only",
        },
    }


def assemble_exclusive_owner_shadow(
    uvlm_load: Any,
    *,
    paired_ldvm_delta_load: Any,
    polar_delta_load: Any,
    persistence: Any,
    paired_ldvm_component_discrepancy: Mapping[str, Any],
    enabled: bool = True,
) -> dict[str, Any]:
    """Assemble the v5d0 three-owner shadow and its two equivalent ledgers.

    Load histories are time-first and use their final axis for load components
    (for example, ``CL`` and ``CD``).  ``persistence`` and every paired-LDVM
    component live on the remaining owner topology.

    When disabled, correction inputs are deliberately not evaluated.  When
    enabled, a nonzero projected LDVM load at a point with a zero component
    witness is rejected because it cannot have arisen from the declared paired
    component ledger and would invalidate exact v4b replay.
    """

    uvlm = _load_history("UVLM load", uvlm_load)
    if not isinstance(enabled, (bool, np.bool_)):
        raise ValueError("enabled must be Boolean")
    if not enabled:
        return _disabled_shadow(uvlm)

    ldvm_delta = _load_history("paired-LDVM delta load", paired_ldvm_delta_load)
    polar_delta = _load_history("polar delta load", polar_delta_load)
    if ldvm_delta.shape != uvlm.shape or polar_delta.shape != uvlm.shape:
        raise ValueError("all load histories must have identical shapes")

    owner_shape = uvlm.shape[:-1]
    material = paired_ldvm_material_witness(paired_ldvm_component_discrepancy)
    if material.shape != owner_shape:
        raise ValueError(
            "paired-LDVM component witness does not match the load owner topology"
        )
    p = _owner_field("persistence", persistence, owner_shape)
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("persistence must lie in the closed interval [0, 1]")

    if np.any((~material)[..., None] & (ldvm_delta != 0.0)):
        raise ValueError(
            "paired-LDVM delta load is nonzero without a component material witness"
        )

    m = material.astype(float)
    w_p = p
    w_l = (1.0 - p) * m
    w_a = (1.0 - p) * (1.0 - m)
    weight_sum = w_a + w_l + w_p

    candidate_a = uvlm
    candidate_l = uvlm + ldvm_delta
    candidate_p = uvlm + polar_delta
    weighted_a = w_a[..., None] * candidate_a
    weighted_l = w_l[..., None] * candidate_l
    weighted_p = w_p[..., None] * candidate_p
    owner_sum = weighted_a + weighted_l + weighted_p

    residual_uvlm = uvlm
    residual_ldvm = w_l[..., None] * ldvm_delta
    residual_polar = w_p[..., None] * polar_delta
    residual_sum = residual_uvlm + residual_ldvm + residual_polar

    # This is the frozen v4b/v5c0 arithmetic written in its original two-way
    # form.  The zero-witness closure above makes it algebraically identical to
    # the exclusive three-owner ledger at every point.
    v4b_replay = (1.0 - p)[..., None] * candidate_l + p[..., None] * candidate_p

    owner_residual_difference = owner_sum - residual_sum
    replay_difference = owner_sum - v4b_replay
    return {
        "load": owner_sum,
        "candidate_loads": {"A": candidate_a, "L": candidate_l, "P": candidate_p},
        "weights": {"wA": w_a, "wL": w_l, "wP": w_p},
        "material_witness": material,
        "persistence": p,
        "weighted_owner_ledger": {
            "A": weighted_a,
            "L": weighted_l,
            "P": weighted_p,
        },
        "residual_ledger": {
            "UVLM": residual_uvlm,
            "paired_LDVM": residual_ldvm,
            "polar": residual_polar,
        },
        "diagnostics": {
            "enabled": True,
            "status": "shadow_only_exact_v4b_replay",
            "max_abs_weight_sum_residual": float(np.max(np.abs(weight_sum - 1.0))),
            "max_abs_owner_residual_ledger_difference": float(
                np.max(np.abs(owner_residual_difference))
            ),
            "max_abs_v4b_replay_difference": float(np.max(np.abs(replay_difference))),
        },
        "model_contract": {
            "shadow_only": True,
            "evaluation_scope": "mechanical_shadow_only",
            "canonical_eligible": False,
            "parameters": {},
            "observation_access": "none",
            "claim_scope": "accounting_equivalence_only",
        },
    }


__all__ = [
    "PAIRED_LDVM_COMPONENTS",
    "assemble_exclusive_owner_shadow",
    "paired_ldvm_material_witness",
]
