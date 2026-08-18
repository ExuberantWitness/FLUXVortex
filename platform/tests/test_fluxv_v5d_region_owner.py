from __future__ import annotations

import numpy as np
import pytest

from forward_flight_benchmarks.fluxv_v5d_region_owner import (
    RegionOwnerParameters,
    cross_section_region_owner,
)


TOLERANCE = 1.0e-15


def test_exact_boundaries_follow_frozen_yang_regions() -> None:
    separation = 0.1
    alpha = np.array(
        [
            [-0.5, -0.1, 0.0, 0.1, 0.5],
            [-np.nextafter(0.5, np.inf), -0.2, 0.2, 0.0, 0.5],
        ]
    )
    result = cross_section_region_owner(
        alpha,
        alpha_sep_rad=separation,
        strip_weights=np.ones(5),
    )
    masks = result["strip_region_masks"]
    np.testing.assert_array_equal(
        masks["A"],
        np.array(
            [[False, True, True, True, False], [False, False, False, True, False]]
        ),
    )
    np.testing.assert_array_equal(
        masks["L"],
        np.array([[True, False, False, False, True], [False, True, True, False, True]]),
    )
    np.testing.assert_array_equal(
        masks["P"],
        np.array(
            [[False, False, False, False, False], [True, False, False, False, False]]
        ),
    )


def test_strip_area_weights_and_per_strip_separation_are_respected() -> None:
    alpha = np.array(
        [
            [0.05, 0.30, 1.10],
            [0.60, 0.20, 0.50],
            [0.00, 0.00, 0.00],
        ]
    )
    result = cross_section_region_owner(
        alpha,
        alpha_sep_rad=np.array([0.1, 0.2, 0.2]),
        strip_weights=np.array([1.0, 2.0, 7.0]),
    )
    np.testing.assert_allclose(result["normalized_strip_weights"], [0.1, 0.2, 0.7])
    np.testing.assert_allclose(result["weights"]["wA"], [0.1, 0.2, 1.0])
    np.testing.assert_allclose(result["weights"]["wL"], [0.2, 0.7, 0.0])
    np.testing.assert_allclose(result["weights"]["wP"], [0.7, 0.1, 0.0])
    summed = sum(result["weights"].values())
    np.testing.assert_allclose(summed, np.ones(3), rtol=0.0, atol=TOLERANCE)
    assert all(np.all(value >= 0.0) for value in result["weights"].values())
    assert result["diagnostics"]["max_abs_weight_sum_residual"] <= TOLERANCE


def test_disabled_is_exact_A_identity_without_numeric_input_evaluation() -> None:
    alpha = np.array([[np.nan, np.inf], [-np.inf, np.nan]])
    result = cross_section_region_owner(
        alpha,
        alpha_sep_rad="not evaluated",
        strip_weights={"not": "evaluated"},
        enabled=False,
    )
    np.testing.assert_array_equal(result["weights"]["wA"], np.ones(2))
    np.testing.assert_array_equal(result["weights"]["wL"], np.zeros(2))
    np.testing.assert_array_equal(result["weights"]["wP"], np.zeros(2))
    np.testing.assert_array_equal(
        result["strip_region_masks"]["A"], np.ones((2, 2), dtype=bool)
    )
    assert result["normalized_strip_weights"] is None
    assert result["alpha_sep_rad_by_strip"] is None
    assert result["diagnostics"]["status"] == "not_evaluated_disabled"


def test_prefix_is_bitwise_unchanged_by_future_alpha() -> None:
    rng = np.random.default_rng(20260814)
    length = 31
    prefix = 17
    alpha = rng.normal(scale=0.4, size=(length, 4))
    first = cross_section_region_owner(
        alpha,
        alpha_sep_rad=np.array([0.08, 0.10, 0.12, 0.14]),
        strip_weights=np.array([1.0, 2.0, 3.0, 4.0]),
    )
    changed_alpha = alpha.copy()
    changed_alpha[prefix:] = rng.normal(scale=100.0, size=(length - prefix, 4))
    changed = cross_section_region_owner(
        changed_alpha,
        alpha_sep_rad=np.array([0.08, 0.10, 0.12, 0.14]),
        strip_weights=np.array([1.0, 2.0, 3.0, 4.0]),
    )
    for key in ("wA", "wL", "wP"):
        np.testing.assert_array_equal(
            first["weights"][key][:prefix], changed["weights"][key][:prefix]
        )
    for key in ("A", "L", "P"):
        np.testing.assert_array_equal(
            first["strip_region_masks"][key][:prefix],
            changed["strip_region_masks"][key][:prefix],
        )


def test_manifest_labels_noncanonical_cross_section_shadow_with_no_force() -> None:
    result = cross_section_region_owner(
        np.zeros((2, 1)), alpha_sep_rad=0.1, strip_weights=np.ones(1)
    )
    assert result["parameters"]["c_alpha"] == 5.0
    assert result["parameters"]["c_alpha_source"] == (
        "Yang et al. (2025), Eqs. (11)--(12)"
    )
    assert result["parameters"]["observation_access"] == "none"
    assert result["parameters"]["case_or_paper_branch"] == "none"
    assert result["model_contract"] == {
        "shadow_only": True,
        "transfer_scope": "cross_section_regions_to_spanwise_area_fractions",
        "canonical_eligible": False,
        "computes_force": False,
        "observation_access": "none",
        "claim_scope": "region-owner mechanics only",
    }


def test_custom_C_alpha_is_rejected() -> None:
    with pytest.raises(ValueError, match="freezes C_alpha=5"):
        RegionOwnerParameters(c_alpha=4.999)


@pytest.mark.parametrize(
    ("alpha", "separation", "weights", "message"),
    [
        (np.array([[0.0], [np.nan]]), 0.1, [1.0], "alpha_rad"),
        (np.ones((2, 1)), np.nan, [1.0], "finite and positive"),
        (np.ones((2, 1)), 0.0, [1.0], "finite and positive"),
        (np.ones((2, 1)), -0.1, [1.0], "finite and positive"),
        (np.ones((2, 2)), [0.1], [1.0, 1.0], "scalar or have shape"),
        (np.ones((2, 2)), 0.1, [1.0, np.inf], "finite and nonnegative"),
        (np.ones((2, 2)), 0.1, [1.0, -1.0], "finite and nonnegative"),
        (np.ones((2, 2)), 0.1, [0.0, 0.0], "at least one"),
        (np.ones((2, 2)), 0.1, [1.0], "shape"),
    ],
)
def test_invalid_numeric_inputs_fail_closed(
    alpha: np.ndarray,
    separation: object,
    weights: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        cross_section_region_owner(
            alpha, alpha_sep_rad=separation, strip_weights=weights
        )


def test_history_topology_and_enabled_flag_are_strict() -> None:
    with pytest.raises(ValueError, match="shape"):
        cross_section_region_owner(
            np.ones(3), alpha_sep_rad=0.1, strip_weights=np.ones(3)
        )
    with pytest.raises(ValueError, match="Boolean"):
        cross_section_region_owner(
            np.ones((2, 1)),
            alpha_sep_rad=0.1,
            strip_weights=np.ones(1),
            enabled=1,
        )


def test_finite_separation_that_overflows_transition_fails_closed() -> None:
    with pytest.raises(ValueError, match=r"C_alpha\*alpha_sep_rad must be finite"):
        cross_section_region_owner(
            np.zeros((1, 1)),
            alpha_sep_rad=np.finfo(float).max,
            strip_weights=np.ones(1),
        )
