from __future__ import annotations

import numpy as np
import pytest

from forward_flight_benchmarks.ldvm_uvlm_correction import (
    LDVMSectionSettings,
    LESPThreshold,
    apply_section_delta_to_coefficients,
    project_ldvm_delta_to_finite_wing,
    run_ldvm_separation_pair,
)


def _threshold(value: float = 0.18) -> LESPThreshold:
    return LESPThreshold(
        value=value,
        section_family="SD7003",
        reynolds=30_000.0,
        source="Ramesh LDVM v2.5 author reference input",
    )


def test_threshold_fails_closed_without_physical_provenance() -> None:
    with pytest.raises(ValueError):
        LESPThreshold(0.18, "", 30_000.0, "")
    with pytest.raises(ValueError):
        LESPThreshold(-0.18, "SD7003", 30_000.0, "source")


def test_no_onset_reduces_exactly_to_baseline() -> None:
    samples = 24
    zero = np.zeros(samples)
    pair = run_ldvm_separation_pair(
        alpha_rad=zero,
        alpha_rate_per_convective_time=zero,
        heave_rate_over_u=zero,
        delta_time_convective=0.02,
        pivot_fraction_chord=0.25,
        threshold=_threshold(),
        settings=LDVMSectionSettings(ndiv=24, naterm=8, max_wake_steps=48),
    )
    assert not np.any(pair["shed_lev"])
    np.testing.assert_array_equal(pair["delta"]["CLf"], zero)
    np.testing.assert_array_equal(pair["delta"]["CDf"], zero)

    baseline_cl = np.linspace(-0.2, 0.3, samples)
    baseline_cd = np.linspace(0.01, 0.05, samples)
    corrected = apply_section_delta_to_coefficients(
        baseline_cl, baseline_cd, pair, finite_wing_gain=0.7
    )
    np.testing.assert_array_equal(corrected["CL"], baseline_cl)
    np.testing.assert_array_equal(corrected["CD"], baseline_cd)


def test_supercritical_history_caps_lesp_and_sheds_material_lev() -> None:
    samples = 80
    alpha = np.deg2rad(np.full(samples, 35.0))
    zero = np.zeros(samples)
    pair = run_ldvm_separation_pair(
        alpha_rad=alpha,
        alpha_rate_per_convective_time=zero,
        heave_rate_over_u=zero,
        delta_time_convective=0.015,
        pivot_fraction_chord=0.0,
        threshold=_threshold(),
        settings=LDVMSectionSettings(ndiv=32, naterm=12, max_wake_steps=100),
    )
    assert np.any(pair["shed_lev"])
    assert pair["lev_count"][-1] > 0
    assert np.max(np.abs(pair["separated"]["A0"])) <= 0.18 + 1.0e-10
    assert np.max(np.abs(pair["delta"]["CLf"])) > 0.01


def test_finite_wing_gain_is_bounded() -> None:
    dummy = {"delta": {"CLf": np.zeros(3), "CDf": np.zeros(3)}}
    with pytest.raises(ValueError):
        apply_section_delta_to_coefficients(
            np.zeros(3), np.zeros(3), dummy, finite_wing_gain=1.1
        )


def test_normal_and_quadratic_suction_use_distinct_finite_wing_gains() -> None:
    alpha = np.array([0.0, np.pi / 2.0])
    result = project_ldvm_delta_to_finite_wing(
        np.ones(2),
        np.zeros(2),
        np.zeros(2),
        np.ones(2),
        alpha,
        aspect_ratio=4.0,
    )
    gain = 2.0 / 3.0
    assert result["normal_gain"] == pytest.approx(gain)
    assert result["axial_suction_gain"] == pytest.approx(gain**2)
    # At alpha=0, suction contributes negative drag; at 90 deg it contributes lift.
    assert result["delta_CL"][0] == pytest.approx(gain)
    assert result["delta_CD"][0] == pytest.approx(-(gain**2))
    assert result["delta_CL"][1] == pytest.approx(gain**2)
    assert result["delta_CD"][1] == pytest.approx(gain)


def test_normal_force_components_keep_separate_finite_span_ledgers() -> None:
    alpha = np.zeros(3)
    result = project_ldvm_delta_to_finite_wing(
        np.ones(3),
        2.0 * np.ones(3),
        3.0 * np.ones(3),
        np.zeros(3),
        alpha,
        aspect_ratio=3.0,
    )
    g = 3.0 / 5.0
    kam = 0.85
    np.testing.assert_allclose(result["projected_delta_CNc"], g)
    np.testing.assert_allclose(result["projected_delta_CNnc"], 2.0 * kam)
    np.testing.assert_allclose(result["projected_delta_CNnonl"], 3.0 * g**2)
    np.testing.assert_allclose(result["delta_CL"], g + 2.0 * kam + 3.0 * g**2)
