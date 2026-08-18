from __future__ import annotations

import numpy as np
import pytest

from forward_flight_benchmarks.fluxv_v5d_owner import (
    assemble_exclusive_owner_shadow,
    paired_ldvm_material_witness,
)


TOLERANCE = 1.0e-12


def _components(*, length: int = 6) -> dict[str, np.ndarray]:
    zeros = np.zeros(length)
    return {
        "CNc": zeros.copy(),
        "CNnc": zeros.copy(),
        "CNnonl": zeros.copy(),
        "CSf": zeros.copy(),
    }


def test_material_witness_is_exact_pointwise_and_parameter_free() -> None:
    components = _components(length=5)
    components["CNc"][1] = np.nextafter(0.0, 1.0)
    components["CNnc"][2] = -0.2
    components["CNnonl"][3] = 0.3
    components["CSf"][4] = -0.4
    np.testing.assert_array_equal(
        paired_ldvm_material_witness(components),
        np.array([False, True, True, True, True]),
    )


def test_disabled_shadow_is_bitwise_uvlm_identity_and_does_not_evaluate_inputs() -> (
    None
):
    uvlm = np.array([[1.0, -0.1], [2.0, 0.2]])
    result = assemble_exclusive_owner_shadow(
        uvlm,
        paired_ldvm_delta_load="not evaluated",
        polar_delta_load={"not": "evaluated"},
        persistence=np.nan,
        paired_ldvm_component_discrepancy={},
        enabled=False,
    )
    np.testing.assert_array_equal(result["load"], uvlm)
    np.testing.assert_array_equal(result["weights"]["wA"], np.ones(2))
    np.testing.assert_array_equal(result["weights"]["wL"], np.zeros(2))
    np.testing.assert_array_equal(result["weights"]["wP"], np.zeros(2))
    assert result["diagnostics"]["status"] == "not_evaluated_disabled"
    assert result["model_contract"]["canonical_eligible"] is False


def test_no_separation_reduces_exactly_to_uvlm() -> None:
    uvlm = np.array([[0.3, -0.1], [0.4, 0.2], [-0.2, 0.5]])
    result = assemble_exclusive_owner_shadow(
        uvlm,
        paired_ldvm_delta_load=np.zeros_like(uvlm),
        polar_delta_load=np.full_like(uvlm, 99.0),
        persistence=np.zeros(3),
        paired_ldvm_component_discrepancy=_components(length=3),
    )
    np.testing.assert_array_equal(result["material_witness"], np.zeros(3, dtype=bool))
    np.testing.assert_array_equal(result["load"], uvlm)
    np.testing.assert_array_equal(result["weights"]["wA"], np.ones(3))


def test_exclusive_weights_ledgers_and_frozen_v4b_replay_close() -> None:
    rng = np.random.default_rng(20260814)
    uvlm = rng.normal(size=(8, 2))
    ldvm_delta = rng.normal(scale=0.1, size=(8, 2))
    polar_delta = rng.normal(scale=0.2, size=(8, 2))
    persistence = np.linspace(0.0, 1.0, 8)
    components = _components(length=8)
    material = np.array([False, True, False, True, True, False, True, False])
    components["CNc"][material] = np.linspace(0.01, 0.04, material.sum())
    ldvm_delta[~material] = 0.0

    result = assemble_exclusive_owner_shadow(
        uvlm,
        paired_ldvm_delta_load=ldvm_delta,
        polar_delta_load=polar_delta,
        persistence=persistence,
        paired_ldvm_component_discrepancy=components,
    )
    weights = result["weights"]
    np.testing.assert_allclose(
        weights["wA"] + weights["wL"] + weights["wP"],
        np.ones(8),
        rtol=0.0,
        atol=TOLERANCE,
    )
    owner_ledger = result["weighted_owner_ledger"]
    residual_ledger = result["residual_ledger"]
    owner_sum = owner_ledger["A"] + owner_ledger["L"] + owner_ledger["P"]
    residual_sum = (
        residual_ledger["UVLM"]
        + residual_ledger["paired_LDVM"]
        + residual_ledger["polar"]
    )
    frozen_v4b = (1.0 - persistence[:, None]) * (uvlm + ldvm_delta) + persistence[
        :, None
    ] * (uvlm + polar_delta)
    np.testing.assert_allclose(result["load"], owner_sum, rtol=0.0, atol=TOLERANCE)
    np.testing.assert_allclose(result["load"], residual_sum, rtol=0.0, atol=TOLERANCE)
    np.testing.assert_allclose(result["load"], frozen_v4b, rtol=0.0, atol=TOLERANCE)
    assert result["diagnostics"]["max_abs_weight_sum_residual"] <= TOLERANCE
    assert (
        result["diagnostics"]["max_abs_owner_residual_ledger_difference"] <= TOLERANCE
    )
    assert result["diagnostics"]["max_abs_v4b_replay_difference"] <= TOLERANCE
    assert result["model_contract"] == {
        "shadow_only": True,
        "evaluation_scope": "mechanical_shadow_only",
        "canonical_eligible": False,
        "parameters": {},
        "observation_access": "none",
        "claim_scope": "accounting_equivalence_only",
    }


def test_zero_component_witness_rejects_an_unowned_ldvm_load() -> None:
    with pytest.raises(ValueError, match="without a component material witness"):
        assemble_exclusive_owner_shadow(
            np.zeros((3, 2)),
            paired_ldvm_delta_load=np.ones((3, 2)),
            polar_delta_load=np.zeros((3, 2)),
            persistence=np.zeros(3),
            paired_ldvm_component_discrepancy=_components(length=3),
        )


def test_shadow_is_causal_under_suffix_changes() -> None:
    length = 12
    prefix = 7
    uvlm = np.column_stack((np.linspace(-1.0, 1.0, length), np.ones(length)))
    ldvm_delta = np.column_stack((np.linspace(0.1, 0.3, length), np.zeros(length)))
    polar_delta = np.column_stack((np.zeros(length), np.linspace(-0.2, 0.2, length)))
    persistence = np.linspace(0.0, 0.9, length)
    components = _components(length=length)
    components["CNc"][:] = np.linspace(0.01, 0.12, length)

    first = assemble_exclusive_owner_shadow(
        uvlm,
        paired_ldvm_delta_load=ldvm_delta,
        polar_delta_load=polar_delta,
        persistence=persistence,
        paired_ldvm_component_discrepancy=components,
    )
    changed_components = {name: value.copy() for name, value in components.items()}
    changed_components["CNc"][prefix:] *= -11.0
    changed = assemble_exclusive_owner_shadow(
        np.concatenate((uvlm[:prefix], 13.0 * uvlm[prefix:])),
        paired_ldvm_delta_load=np.concatenate(
            (ldvm_delta[:prefix], -17.0 * ldvm_delta[prefix:])
        ),
        polar_delta_load=np.concatenate(
            (polar_delta[:prefix], 19.0 * polar_delta[prefix:])
        ),
        persistence=np.concatenate((persistence[:prefix], 1.0 - persistence[prefix:])),
        paired_ldvm_component_discrepancy=changed_components,
    )
    np.testing.assert_array_equal(first["load"][:prefix], changed["load"][:prefix])
    np.testing.assert_array_equal(
        first["material_witness"][:prefix], changed["material_witness"][:prefix]
    )


def test_component_shape_and_nonfinite_values_fail_closed() -> None:
    components = _components(length=3)
    components["CSf"] = np.zeros(4)
    with pytest.raises(ValueError, match="identical shapes"):
        paired_ldvm_material_witness(components)

    components = _components(length=3)
    components["CNc"][1] = np.nan
    with pytest.raises(ValueError, match="must be finite"):
        paired_ldvm_material_witness(components)
