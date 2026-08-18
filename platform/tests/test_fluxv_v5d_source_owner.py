from __future__ import annotations

import numpy as np
import pytest

from forward_flight_benchmarks.fluxv_v5d_source_owner import (
    SourceTimeOwnerParameters,
    source_time_causal_persistence,
)


def _manual_cascade(
    alpha: np.ndarray, delta_t_tilde: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    signed_fast = np.zeros(alpha.shape[1])
    signed_slow = np.zeros(alpha.shape[1])
    magnitude_fast = np.zeros(alpha.shape[1])
    magnitude_slow = np.zeros(alpha.shape[1])
    histories = tuple(np.zeros_like(alpha) for _ in range(4))
    for index, (current, step) in enumerate(zip(alpha, delta_t_tilde, strict=True)):
        fast_gain = 1.0 - np.exp(-0.30 * step)
        slow_gain = 1.0 - np.exp(-0.045 * step)
        signed_fast += fast_gain * (current - signed_fast)
        magnitude_fast += fast_gain * (np.abs(current) - magnitude_fast)
        signed_slow += slow_gain * (signed_fast - signed_slow)
        magnitude_slow += slow_gain * (magnitude_fast - magnitude_slow)
        for history, value in zip(
            histories,
            (signed_fast, signed_slow, magnitude_fast, magnitude_slow),
            strict=True,
        ):
            history[index] = value
    return histories


def test_manifest_freezes_source_poles_clock_units_and_scope() -> None:
    manifest = SourceTimeOwnerParameters().manifest()
    assert manifest["fast_pole_per_t_tilde"] == 0.30
    assert manifest["slow_pole_per_t_tilde"] == 0.045
    assert manifest["delta_t_tilde_definition"] == (
        "2*|V_rel at 0.75c, perpendicular to local span|*dt/c_local"
    )
    assert manifest["observation_fit"] == "none"
    assert manifest["case_or_paper_branch"] == "none"
    with pytest.raises(ValueError, match="source-frozen"):
        SourceTimeOwnerParameters(fast_pole_per_t_tilde=0.31)


def test_local_source_time_matches_independent_stepwise_exponential_cascade() -> None:
    alpha = np.array([[0.3, -0.2], [-0.1, 0.5], [0.4, -0.7], [-0.2, 0.1]], dtype=float)
    delta_t_tilde = np.array(
        [[0.01, 0.40], [0.03, 0.20], [0.50, 0.02], [0.07, 0.30]], dtype=float
    )
    result = source_time_causal_persistence(
        alpha,
        delta_t_tilde=delta_t_tilde,
        strip_weights=np.array([1.0, 3.0]),
    )
    expected = _manual_cascade(alpha, delta_t_tilde)
    for key, history in zip(
        (
            "signed_fast_history_rad",
            "signed_slow_history_rad",
            "magnitude_fast_history_rad",
            "magnitude_slow_history_rad",
        ),
        expected,
        strict=True,
    ):
        np.testing.assert_allclose(result[key], history, rtol=2.0e-14, atol=1.0e-16)
    np.testing.assert_allclose(result["normalized_strip_weights"], [0.25, 0.75])
    np.testing.assert_allclose(
        result["global_persistence"],
        np.asarray(result["strip_persistence"]) @ np.array([0.25, 0.75]),
    )


def test_disabled_is_exact_zero_and_does_not_evaluate_numeric_inputs() -> None:
    alpha = np.array([[np.nan, np.inf], [-np.inf, np.nan]])
    result = source_time_causal_persistence(
        alpha,
        delta_t_tilde="not evaluated",
        strip_weights={"not": "evaluated"},
        enabled=False,
    )
    np.testing.assert_array_equal(result["global_persistence"], np.zeros(2))
    np.testing.assert_array_equal(result["strip_persistence"], np.zeros((2, 2)))
    np.testing.assert_array_equal(result["signed_slow_history_rad"], np.zeros((2, 2)))
    assert result["normalized_strip_weights"] is None
    assert result["diagnostics"] == {
        "enabled": False,
        "status": "not_evaluated_disabled",
        "state_updated": False,
        "causal": True,
        "lookahead_samples": 0,
    }


def test_prefix_is_bitwise_unchanged_by_every_suffix_input() -> None:
    rng = np.random.default_rng(20260814)
    length = 40
    prefix = 23
    alpha = rng.normal(scale=0.2, size=(length, 4))
    delta_t_tilde = rng.uniform(0.005, 0.4, size=(length, 4))
    first = source_time_causal_persistence(
        alpha,
        delta_t_tilde=delta_t_tilde,
        strip_weights=np.array([1.0, 2.0, 3.0, 4.0]),
    )
    changed_alpha = alpha.copy()
    changed_alpha[prefix:] = rng.normal(scale=100.0, size=(length - prefix, 4))
    changed_delta = delta_t_tilde.copy()
    changed_delta[prefix:] = rng.uniform(2.0, 20.0, size=(length - prefix, 4))
    changed = source_time_causal_persistence(
        changed_alpha,
        delta_t_tilde=changed_delta,
        strip_weights=np.array([1.0, 2.0, 3.0, 4.0]),
    )
    for key in (
        "global_persistence",
        "strip_persistence",
        "signed_fast_history_rad",
        "signed_slow_history_rad",
        "magnitude_fast_history_rad",
        "magnitude_slow_history_rad",
    ):
        np.testing.assert_array_equal(first[key][:prefix], changed[key][:prefix])


def test_zero_and_one_sign_limits_are_bounded_and_exact() -> None:
    steps = np.full((30, 3), 0.1)
    zero = source_time_causal_persistence(np.zeros((30, 3)), delta_t_tilde=steps)
    np.testing.assert_array_equal(zero["strip_persistence"], np.zeros((30, 3)))
    positive = source_time_causal_persistence(
        np.full((30, 3), 0.2), delta_t_tilde=steps
    )
    np.testing.assert_array_equal(positive["strip_persistence"], np.ones((30, 3)))
    assert np.all(np.asarray(positive["global_persistence"]) <= 1.0)
    assert np.all(np.asarray(positive["global_persistence"]) >= 0.0)


@pytest.mark.parametrize(
    ("alpha", "delta_t_tilde", "weights", "message"),
    [
        (np.array([[0.1], [np.nan]]), np.ones((2, 1)), None, "alpha_rad"),
        (np.ones((2, 1)), np.array([[0.1], [np.inf]]), None, "finite and positive"),
        (np.ones((2, 1)), np.array([[0.1], [0.0]]), None, "finite and positive"),
        (np.ones((2, 1)), np.array([[0.1], [-0.1]]), None, "finite and positive"),
        (np.ones((2, 2)), np.ones((2, 2)), [1.0, np.nan], "finite"),
        (np.ones((2, 2)), np.ones((2, 2)), [0.0, 0.0], "positive"),
    ],
)
def test_nonfinite_nonpositive_and_invalid_weights_fail_closed(
    alpha: np.ndarray,
    delta_t_tilde: np.ndarray,
    weights: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        source_time_causal_persistence(
            alpha, delta_t_tilde=delta_t_tilde, strip_weights=weights
        )


def test_time_strip_topology_is_strict() -> None:
    with pytest.raises(ValueError, match="shape"):
        source_time_causal_persistence(np.ones(4), delta_t_tilde=np.ones(4))
    with pytest.raises(ValueError, match="match alpha_rad"):
        source_time_causal_persistence(np.ones((4, 2)), delta_t_tilde=np.ones((4, 1)))


def test_finite_but_ill_scaled_history_fails_closed() -> None:
    maximum = np.finfo(float).max
    with pytest.raises(ValueError, match="state.*not finite"):
        source_time_causal_persistence(
            np.array([[maximum], [-maximum]]),
            delta_t_tilde=np.full((2, 1), 100.0),
        )
