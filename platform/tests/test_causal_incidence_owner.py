from __future__ import annotations

import numpy as np

from forward_flight_benchmarks.causal_incidence_owner import (
    causal_incidence_persistence,
)


def test_sustained_incidence_converges_to_persistent_owner() -> None:
    alpha = np.full((2000, 2), np.deg2rad(10.0))
    result = causal_incidence_persistence(alpha, delta_time_convective=0.1)
    assert result["global_persistence"][-1] > 0.999


def test_zero_mean_fast_oscillation_selects_transient_owner() -> None:
    phase = np.arange(6000) * 0.05
    alpha = np.deg2rad(20.0) * np.sin(phase)
    result = causal_incidence_persistence(alpha, delta_time_convective=0.05)
    tail = np.asarray(result["global_persistence"])[-1000:]
    assert float(np.mean(tail)) < 0.05


def test_causality_prefix_is_unchanged_by_future_samples() -> None:
    rng = np.random.default_rng(4)
    prefix = rng.normal(size=(80, 3)) * 0.1
    history_a = np.vstack((prefix, np.ones((30, 3))))
    history_b = np.vstack((prefix, -np.ones((30, 3))))
    result_a = causal_incidence_persistence(history_a, delta_time_convective=0.2)
    result_b = causal_incidence_persistence(history_b, delta_time_convective=0.2)
    np.testing.assert_array_equal(
        result_a["strip_persistence"][:80],
        result_b["strip_persistence"][:80],
    )


def test_zero_incidence_is_finite_and_zero() -> None:
    result = causal_incidence_persistence(np.zeros((20, 1)), delta_time_convective=0.1)
    np.testing.assert_array_equal(result["global_persistence"], np.zeros(20))
