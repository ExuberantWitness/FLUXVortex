from __future__ import annotations

import numpy as np

from forward_flight_benchmarks.razak2014 import (
    RAZAK_2014_CASES,
    build_razak_movement,
)


def test_razak_case_matrix_and_thickness_scope() -> None:
    assert set(RAZAK_2014_CASES) == {9, 10, 11, 13, 14, 15}
    case = RAZAK_2014_CASES[9]
    assert np.isclose(case.area_m2, 0.064)
    assert np.isclose(case.aspect_ratio, 2.5)
    assert np.isclose(case.pitch_center_deg, 2.0)
    assert np.isclose(case.pitch_amplitude_deg, 6.0)
    assert case.pitch_phase_deg == 90.0
    assert RAZAK_2014_CASES[13].pitch_phase_deg == -90.0


def test_razak_smoke_geometry_preserves_hinge_offset() -> None:
    case = RAZAK_2014_CASES[9]
    movement, metadata = build_razak_movement(case, "smoke")
    assert metadata["grid_chord_span"] == [3, 5]
    assert metadata["steps_per_cycle"] == 32
    first = movement.airplanes[0][0].wings[0]
    panels = first.panels
    assert panels.shape == (3, 5)
    root_le = np.asarray(panels[0, 0].Flpp_G_Cg, dtype=float)
    root_te = np.asarray(panels[-1, 0].Blpp_G_Cg, dtype=float)
    tip_le = np.asarray(panels[0, -1].Frpp_G_Cg, dtype=float)
    tip_te = np.asarray(panels[-1, -1].Brpp_G_Cg, dtype=float)
    root_pivot = root_le + case.pitch_axis_fraction_chord * (root_te - root_le)
    tip_pivot = tip_le + case.pitch_axis_fraction_chord * (tip_te - tip_le)
    assert np.isclose(np.linalg.norm(root_pivot[[1, 2]]), case.root_offset_m)
    assert np.isclose(
        np.linalg.norm(tip_pivot[[1, 2]]), case.root_offset_m + case.span_m
    )


def test_razak_nominal_pitch_lead_and_lag_extrema() -> None:
    for figure, expected_at_zero in ((9, 8.0), (13, -5.0)):
        case = RAZAK_2014_CASES[figure]
        phase = np.arange(128, dtype=float) / 128.0
        pitch = case.pitch_center_deg + case.pitch_amplitude_deg * np.sin(
            2.0 * np.pi * phase + np.deg2rad(case.pitch_phase_deg)
        )
        assert np.isclose(pitch[0], expected_at_zero)
        assert np.isclose(np.min(pitch), case.pitch_min_deg)
        assert np.isclose(np.max(pitch), case.pitch_max_deg)
