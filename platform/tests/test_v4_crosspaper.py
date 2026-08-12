from __future__ import annotations

import numpy as np

from forward_flight_benchmarks.run_v4_crosspaper import (
    _fig14_effective_alpha,
    _periodic_resample,
    _yang_kinematics,
)


def test_yang_nondimensional_surface_velocity_has_large_flap_incidence() -> None:
    history = _yang_kinematics(0.0, steps_per_cycle=128, cycles=1, strip_count=4)
    alpha_deg = np.rad2deg(history["effective_alpha_rad"])
    # This catches the dimensional bug U-[dphi/dt*]*r instead of
    # 1-[dphi/d(tU/c)]*(r/c), which reduced the motion to <1 degree.
    assert float(np.max(alpha_deg)) > 20.0
    assert float(np.min(alpha_deg)) < -20.0
    assert abs(float(np.mean(alpha_deg))) < 0.2


def test_fig14_effective_incidence_is_half_cycle_antisymmetric() -> None:
    phase = np.arange(128) * 2.0 * np.pi / 128
    alpha = _fig14_effective_alpha(15.0, 45.0, phase)
    np.testing.assert_allclose(alpha[:64], -alpha[64:], atol=2.0e-15, rtol=0.0)


def test_periodic_resample_preserves_constant_and_phase() -> None:
    np.testing.assert_array_equal(_periodic_resample(np.ones(17), 64), np.ones(64))
    source = np.sin(2.0 * np.pi * np.arange(32) / 32)
    target = _periodic_resample(source, 64)
    assert int(np.argmax(target)) == 16
