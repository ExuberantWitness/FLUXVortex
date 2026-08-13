"""Regression tests for the Baik 2012 W1--W4 reconstruction contract."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
import pytest

from forward_flight_benchmarks.baik2012 import (
    BAIK_2012_CASES,
    baik_heave_spacing,
    baik_kinematics,
    build_baik_movement,
    sharp_fourier_lowpass,
)
from forward_flight_benchmarks.uvlm_polar_correction import movement_polar_residual


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_CSV = (
    REPO_ROOT
    / "docs/forward_flight_large_pitch/reproductions/baik2012_w1_w4/source_data"
    / "baik2012_w1_w4_corrected_total_cl_cd.csv"
)
SOURCE_SHA256 = "4de6b01cd8072959e5b780053f311efa92ab5a94f17940dd122df340ad638f2f"


def test_w1_w4_source_matrix_and_w3_typo_resolution() -> None:
    assert tuple(BAIK_2012_CASES) == ("W1", "W2", "W3", "W4")
    assert BAIK_2012_CASES["W3"].reduced_frequency == 1.0
    assert BAIK_2012_CASES["W3"].heave_to_chord == 0.25
    assert BAIK_2012_CASES["W4"].heave_to_chord == 1.0
    assert BAIK_2012_CASES["W1"].kinematic_strouhal == pytest.approx(0.5 / np.pi)
    assert BAIK_2012_CASES["W2"].kinematic_strouhal == pytest.approx(1.0 / np.pi)
    assert BAIK_2012_CASES["W1"].experimental_filter_harmonic == 7
    assert BAIK_2012_CASES["W2"].experimental_filter_harmonic == 3
    assert BAIK_2012_CASES["W1"].rounded_edge_radius_m == pytest.approx(0.002375)


@pytest.mark.parametrize("case_id", tuple(BAIK_2012_CASES))
def test_source_effective_incidence_and_pitch_phase(case_id: str) -> None:
    case = BAIK_2012_CASES[case_id]
    phase = np.asarray([0.0, 0.25, 0.5, 0.75])
    motion = baik_kinematics(phase, case)
    np.testing.assert_allclose(
        motion["effective_alpha_deg"], [8.0, 22.0, 8.0, -6.0], atol=1.0e-11
    )
    np.testing.assert_allclose(
        motion["geometric_alpha_deg"],
        [
            8.0,
            8.0 - case.implemented_pitch_amplitude_deg,
            8.0,
            8.0 + case.implemented_pitch_amplitude_deg,
        ],
        atol=1.0e-11,
    )


@pytest.mark.parametrize("case_id", ("W1", "W2"))
def test_nonlinear_heave_spacing_contract(case_id: str) -> None:
    spacing = baik_heave_spacing(case_id)
    phase_rad = np.asarray([0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi, 2.0 * np.pi])
    np.testing.assert_allclose(
        spacing(phase_rad), [0.0, 1.0, 0.0, -1.0, 0.0], atol=3.0e-12
    )
    dense = spacing(np.linspace(0.0, 2.0 * np.pi, 1024, endpoint=False))
    assert np.max(dense) == pytest.approx(1.0, abs=2.0e-7)
    assert np.min(dense) == pytest.approx(-1.0, abs=2.0e-7)


def test_ptera_motion_rotates_about_quarter_chord_and_starts_at_upper_limit() -> None:
    case = BAIK_2012_CASES["W1"]
    movement, metadata = build_baik_movement(
        case,
        settings=(2, 4, 32, 1),
    )
    expected_z = case.heave_to_chord * case.chord_m * np.asarray([1.0, 0.0, -1.0, 0.0])
    expected_pitch = baik_kinematics(np.asarray([0.0, 0.25, 0.5, 0.75]), case)[
        "geometric_alpha_deg"
    ]
    pivot_history = []
    for index, (z_value, pitch_value) in enumerate(
        zip(expected_z, expected_pitch, strict=True)
    ):
        wing = movement.airplanes[0][8 * index].wings[0]
        leading_panel = wing.panels[0, 0]
        trailing_panel = wing.panels[-1, 0]
        root_quarter_chord = leading_panel.Flpp_G_Cg + 0.25 * (
            trailing_panel.Blpp_G_Cg - leading_panel.Flpp_G_Cg
        )
        pivot_history.append(root_quarter_chord)
        assert root_quarter_chord[2] == pytest.approx(z_value, abs=5.0e-12)
        assert wing.angles_Gs_to_Wn_ixyz[1] == pytest.approx(pitch_value, abs=5.0e-12)
    assert metadata["grid_chord_span"] == [2, 4]
    pivot_history = np.asarray(pivot_history)
    np.testing.assert_allclose(pivot_history[:, :2], 0.0, atol=2.0e-14)


def test_ptera_geometry_recovers_declared_effective_incidence() -> None:
    case = BAIK_2012_CASES["W2"]
    movement, _metadata = build_baik_movement(
        case,
        settings=(4, 8, 128, 1),
    )
    residual = movement_polar_residual(
        movement,
        source_cycle_step_range=[0, 127],
        period_s=case.period_s,
        freestream_m_s=case.freestream_m_s,
        rho_kg_m3=case.rho_kg_m3,
        aspect_ratio=case.geometric_aspect_ratio,
        output_samples=128,
    )
    geometric_alpha = np.rad2deg(np.asarray(residual["alpha_rad"], dtype=float))
    analytic_alpha = baik_kinematics(np.arange(128, dtype=float) / 128.0, case)[
        "effective_alpha_deg"
    ]
    # This compares a velocity reconstructed from the moving Ptera geometry
    # against the independent analytic law, checking phase, pivot and signs.
    np.testing.assert_allclose(
        geometric_alpha,
        np.broadcast_to(analytic_alpha[:, None], geometric_alpha.shape),
        atol=0.03,
    )


def test_sharp_fourier_lowpass_preserves_allowed_harmonics_and_mean() -> None:
    phase = np.arange(128, dtype=float) / 128
    signal = 2.0 + np.sin(2.0 * np.pi * phase) + 0.3 * np.cos(10.0 * np.pi * phase)
    filtered = sharp_fourier_lowpass(signal, maximum_harmonic=3)
    expected = 2.0 + np.sin(2.0 * np.pi * phase)
    np.testing.assert_allclose(filtered, expected, atol=2.0e-14)
    assert np.mean(filtered) == pytest.approx(np.mean(signal), abs=2.0e-14)


def test_corrected_total_source_hash_and_printed_mean_landmarks() -> None:
    assert hashlib.sha256(SOURCE_CSV.read_bytes()).hexdigest() == SOURCE_SHA256
    with SOURCE_CSV.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    printed = {
        "W1": (1.04, 0.0315),
        "W2": (2.11, -0.127),
        "W3": (1.14, 0.127),
        "W4": (1.37, -0.308),
    }
    for case_id, (printed_cl, printed_cd) in printed.items():
        selected = [row for row in rows if row["case"] == case_id]
        phase = np.asarray([float(row["phase_t_over_T"]) for row in selected])
        cl = np.asarray([float(row["cl"]) for row in selected])
        cd = np.asarray([float(row["cd"]) for row in selected])
        assert len(selected) == 401
        # Raster extraction is never shifted to force agreement with these
        # independent source landmarks. W2 CL has a declared 0.03 tolerance
        # because six coefficient units occupy only 408 vertical pixels.
        cl_tolerance = 0.03 if case_id == "W2" else 0.02
        assert np.trapezoid(cl, phase) == pytest.approx(printed_cl, abs=cl_tolerance)
        assert np.trapezoid(cd, phase) == pytest.approx(printed_cd, abs=0.02)
