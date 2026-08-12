from __future__ import annotations

import numpy as np

from forward_flight_benchmarks.periodic_load_ownership import (
    blend_periodic_persistent_owner,
    periodic_incidence_persistence,
)
from forward_flight_benchmarks.cases import YANG_2025
from forward_flight_benchmarks.ptera_adapter import build_yang2025_movement
from forward_flight_benchmarks.uvlm_polar_correction import movement_polar_residual


def test_symmetric_reversing_incidence_has_zero_persistence() -> None:
    phase = np.arange(64, dtype=float) / 64.0
    alpha = np.sin(2.0 * np.pi * phase)[:, None] * np.array([[0.2, 0.5]])
    global_fraction, strip_fraction = periodic_incidence_persistence(alpha)
    np.testing.assert_allclose(strip_fraction, 0.0, atol=1.0e-15)
    assert np.isclose(global_fraction, 0.0, atol=1.0e-15)


def test_steady_incidence_has_unit_persistence() -> None:
    alpha = np.full((16, 3), 0.3)
    global_fraction, strip_fraction = periodic_incidence_persistence(
        alpha, strip_weights=np.array([1.0, 2.0, 3.0])
    )
    np.testing.assert_allclose(strip_fraction, 1.0)
    assert global_fraction == 1.0


def test_real_yang_resampled_strip_ledger_is_self_consistent() -> None:
    movement, _ = build_yang2025_movement(0.0, "smoke")
    raw = movement_polar_residual(
        movement,
        source_cycle_step_range=(20, 39),
        period_s=YANG_2025.period_s,
        freestream_m_s=YANG_2025.freestream_m_s,
        rho_kg_m3=YANG_2025.rho_kg_m3,
        aspect_ratio=YANG_2025.aspect_ratio,
        output_samples=32,
    )
    total = np.asarray(raw["delta_force_g_n"])
    strip = np.asarray(raw["strip_delta_force_g_n"])
    strip_mean = np.asarray(raw["mean_strip_delta_force_g_n"])
    np.testing.assert_allclose(np.sum(strip, axis=1), total, atol=1.0e-14)
    np.testing.assert_allclose(np.mean(strip, axis=0), strip_mean, atol=1.0e-14)
    np.testing.assert_allclose(
        np.mean(total, axis=0),
        raw["mean_delta_force_g_n"],
        atol=1.0e-14,
    )


def test_persistent_owner_exact_reductions_and_mean_contract() -> None:
    phase = np.arange(16, dtype=float) / 16.0
    attached = {
        "phase": phase,
        "lift_n": 2.0 + np.sin(2.0 * np.pi * phase),
        "drag_n": -1.0 + np.cos(2.0 * np.pi * phase),
        "mean_lift_n": 2.0,
        "mean_drag_n": -1.0,
    }
    separated = {
        "phase": phase,
        "lift_n": 8.0 + 3.0 * np.sin(2.0 * np.pi * phase),
        "drag_n": 4.0 + 2.0 * np.cos(2.0 * np.pi * phase),
        "mean_lift_n": 8.0,
        "mean_drag_n": 4.0,
    }
    common = {
        "attached_history": attached,
        "separated_history": separated,
        "separation_fraction": np.linspace(0.0, 1.0, phase.size),
        "rho_kg_m3": 1.0,
        "freestream_m_s": 2.0,
        "area_m2": 0.5,
    }
    zero = blend_periodic_persistent_owner(**common, persistence_fraction=0.0)
    np.testing.assert_allclose(zero["lift_n"], attached["lift_n"])
    np.testing.assert_allclose(zero["drag_n"], attached["drag_n"])
    assert zero["mean_lift_n"] == 2.0
    assert zero["mean_drag_n"] == -1.0

    half = blend_periodic_persistent_owner(**common, persistence_fraction=0.5)
    assert half["mean_lift_n"] == 5.0
    assert half["mean_drag_n"] == 1.5
    np.testing.assert_allclose(np.mean(half["lift_n"]), 5.0)
    np.testing.assert_allclose(np.mean(half["drag_n"]), 1.5)
