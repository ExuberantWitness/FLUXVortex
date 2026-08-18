"""Fast contract tests for the FluxV v5b force-promotion gate."""

from __future__ import annotations

import numpy as np

from forward_flight_benchmarks.run_fluxv_v5b_force_gate import (
    _smooth_ramp_geometry_velocity,
)


def test_smooth_ramp_has_stationary_endpoints_and_finite_geometry() -> None:
    start_corners, start_velocity = _smooth_ramp_geometry_velocity(0.0)
    end_corners, end_velocity = _smooth_ramp_geometry_velocity(0.2)

    assert start_corners.shape == (3, 3, 3)
    assert end_corners.shape == start_corners.shape
    assert np.all(np.isfinite(start_corners))
    assert np.all(np.isfinite(end_corners))
    assert np.array_equal(start_velocity, np.zeros_like(start_velocity))
    assert np.array_equal(end_velocity, np.zeros_like(end_velocity))
    assert not np.array_equal(start_corners, end_corners)


def test_smooth_ramp_velocity_is_nonzero_only_inside_transition() -> None:
    _, before = _smooth_ramp_geometry_velocity(-0.01)
    _, middle = _smooth_ramp_geometry_velocity(0.1)
    _, after = _smooth_ramp_geometry_velocity(0.21)

    assert np.array_equal(before, np.zeros_like(before))
    assert np.max(np.abs(middle)) > 0.0
    assert np.array_equal(after, np.zeros_like(after))
