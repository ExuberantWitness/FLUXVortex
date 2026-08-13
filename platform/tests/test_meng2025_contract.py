from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from forward_flight_benchmarks.meng2025_case import (
    MENG_2025,
    balance_to_wind_axis,
    build_meng2025_movement,
    nominal_kinematics_deg,
)


SOURCE_DATA = (
    Path(__file__).resolve().parents[1]
    / "forward_flight_benchmarks/source_data/meng2025_fig16_mean_lines_digitized.csv"
)


def test_published_amplitudes_are_peak_to_peak_and_phase_is_downstroke_nose_down() -> (
    None
):
    phase = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0])
    flap, pitch = nominal_kinematics_deg(phase, 22.5)
    np.testing.assert_allclose(flap, [22.5, 0.0, -22.5, 0.0, 22.5], atol=1.0e-12)
    np.testing.assert_allclose(pitch, [0.0, -11.25, 0.0, 11.25, 0.0], atol=1.0e-12)


def test_planform_adapter_obeys_printed_dimensions_and_area() -> None:
    case = MENG_2025
    chord = case.chord_m(
        np.asarray([0.0, case.constant_chord_span_m, case.half_span_m])
    )
    np.testing.assert_allclose(chord, [0.287, 0.287, 0.0], atol=1.0e-15)
    expected = 2.0 * 0.287 * (0.340 + np.pi * (0.800 - 0.340) / 4.0)
    assert np.isclose(case.area_m2, expected)
    assert 6.3 < case.aspect_ratio < 6.4


def test_equation_11_gravity_rotates_to_lift_only() -> None:
    fx = np.asarray([-2.0, 0.5, 3.0])
    fz = np.asarray([1.0, -4.0, 0.25])
    lift_zero_g, thrust_zero_g = balance_to_wind_axis(
        fx, fz, alpha_deg=13.0, gravity_force=0.0
    )
    lift, thrust = balance_to_wind_axis(fx, fz, alpha_deg=13.0, gravity_force=42.0)
    np.testing.assert_allclose(lift - lift_zero_g, 42.0, atol=1.0e-13)
    np.testing.assert_allclose(thrust, thrust_zero_g, atol=1.0e-13)


def test_movement_uses_custom_thin_membrane_not_naca_thickness() -> None:
    movement, metadata = build_meng2025_movement(22.5, quality="smoke")
    assert metadata["pitch_half_amplitude_deg"] == 11.25
    assert metadata["flap_half_amplitude_deg"] == 22.5
    assert "NACA" not in metadata["airfoil_adapter"].upper()
    assert metadata["requested_steps_per_cycle"] == 24
    assert np.isclose(MENG_2025.period_s / movement.delta_time, 24.0)
    right_wing_angles = np.asarray(
        [
            movement.airplanes[0][step].wings[0].angles_Gs_to_Wn_ixyz
            for step in (0, 6, 12, 18)
        ]
    )
    np.testing.assert_allclose(
        right_wing_angles[:, :2],
        [[22.5, 5.0], [0.0, -6.25], [-22.5, 5.0], [0.0, 16.25]],
        atol=1.0e-12,
    )


def test_figure16_semantic_mapping_matches_section_42_text() -> None:
    with SOURCE_DATA.open(newline="", encoding="utf-8") as stream:
        rows = {
            float(row["twist_amplitude_peak_to_peak_deg"]): row
            for row in csv.DictReader(stream)
        }
    lift = {key: float(row["mean_lift_gf"]) for key, row in rows.items()}
    thrust = {key: float(row["mean_net_thrust_gf"]) for key, row in rows.items()}
    assert abs(lift[0.0] - lift[22.5]) < 50.0
    assert lift[45.0] < lift[0.0] and lift[45.0] < lift[22.5]
    assert thrust[22.5] > thrust[0.0] and thrust[22.5] > thrust[45.0]
