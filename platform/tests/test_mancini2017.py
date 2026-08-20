from __future__ import annotations

import numpy as np
import pytest

import forward_flight_benchmarks.mancini2017 as mancini_module
from forward_flight_benchmarks.mancini2017 import (
    FROZEN_V4B_LESP_CRITICAL,
    MANCINI_2017_CASES,
    MANCINI_EXPERIMENT_HAS_SPANWISE_LOADS,
    apply_frozen_mancini_v4b,
    build_mancini_movement,
    load_mancini_fig4_13b_experiment,
    mancini_periodic_pitch_spacing,
    mancini_pitch_angle_deg,
)


def test_geometry_and_published_motion_matrix_are_frozen() -> None:
    fast = MANCINI_2017_CASES["fast_pitch"]
    slow = MANCINI_2017_CASES["slow_pitch"]
    for case in (fast, slow):
        assert case.chord_m == 0.0762
        assert case.span_m == pytest.approx(0.3048)
        assert case.area_m2 == pytest.approx(0.02322576)
        assert case.aspect_ratio == 4.0
        assert case.thickness_fraction == 0.05
        assert case.freestream_m_s == 0.26
        assert case.reynolds == 20_000.0
        assert case.pivot_fraction_chord == 0.0
        assert case.initial_pitch_deg == 0.0
        assert case.maximum_pitch_deg == 45.0
    assert fast.acceleration_distance_chords == 1.0
    assert fast.reduced_pitch_rate == 0.39
    assert fast.ideal_reduced_pitch_rate == pytest.approx(np.pi / 8.0)
    assert fast.eldredge_smoothing == 15.0
    assert slow.acceleration_distance_chords == 6.0
    assert slow.reduced_pitch_rate == 0.065
    assert slow.ideal_reduced_pitch_rate == pytest.approx(np.pi / 48.0)
    assert slow.eldredge_smoothing == 4.0


@pytest.mark.parametrize("case_id", tuple(MANCINI_2017_CASES))
def test_eldredge_motion_has_published_midpoint_and_distant_hold(case_id: str) -> None:
    case = MANCINI_2017_CASES[case_id]
    duration = case.acceleration_distance_chords
    values = mancini_pitch_angle_deg(
        np.asarray((-4.0, 0.5 * duration, duration + 4.0)), case
    )
    assert values[0] < 1.0e-8
    assert values[1] == pytest.approx(22.5, abs=1.0e-11)
    assert values[2] > 45.0 - 1.0e-8

    midpoint = 0.5 * duration
    epsilon = 1.0e-5
    slope = (
        mancini_pitch_angle_deg(midpoint + epsilon, case)
        - mancini_pitch_angle_deg(midpoint - epsilon, case)
    ) / (2.0 * epsilon)
    recovered_k = np.deg2rad(float(slope)) / 2.0
    assert recovered_k == pytest.approx(case.ideal_reduced_pitch_rate, rel=2.0e-5)


@pytest.mark.parametrize("case_id", tuple(MANCINI_2017_CASES))
def test_periodic_carrier_matches_physical_motion_through_scored_window(
    case_id: str,
) -> None:
    case = MANCINI_2017_CASES[case_id]
    scored = np.linspace(0.0, case.observation_chords, 101)
    global_time = case.warmup_chords + scored
    phase = 2.0 * np.pi * global_time / case.waveform_period_chords
    carrier = mancini_periodic_pitch_spacing(phase, case)
    expected = mancini_pitch_angle_deg(scored, case) / case.maximum_pitch_deg
    np.testing.assert_allclose(carrier, expected, rtol=0.0, atol=2.0e-14)
    assert case.closure_start_chords > case.warmup_chords + case.observation_chords


def test_digitized_experiment_is_whole_wing_lift_only_and_has_curve_landmarks() -> None:
    experiment = load_mancini_fig4_13b_experiment()
    assert experiment["data_role"] == "digitized_experimental_whole_wing_lift_curve"
    assert experiment["observed_channels"] == ("CL",)
    assert experiment["spanwise_loads_available"] is False
    assert MANCINI_EXPERIMENT_HAS_SPANWISE_LOADS is False
    assert experiment["t_star"].shape == (1301,)
    fast_peak = int(np.argmax(experiment["CL_fast_pitch"]))
    slow_peak = int(np.argmax(experiment["CL_slow_pitch"]))
    assert experiment["CL_fast_pitch"][fast_peak] == pytest.approx(4.9, abs=0.25)
    assert experiment["t_star"][fast_peak] <= 1.1
    assert experiment["CL_slow_pitch"][slow_peak] == pytest.approx(2.05, abs=0.15)
    assert 4.0 <= experiment["t_star"][slow_peak] <= 6.0


def test_movement_metadata_marks_chordwise_spanwise_and_unresolved_thickness() -> None:
    case = MANCINI_2017_CASES["fast_pitch"]
    movement, metadata = build_mancini_movement(case, "smoke")
    assert metadata["grid_chord_semispan"] == [2, 6]
    assert metadata["steps_per_chord"] == 24
    assert metadata["pitch_trigger_step"] == 48
    assert metadata["span_m"] == pytest.approx(case.span_m)
    assert "complete AR4 rectangle" in metadata["span_adapter"]
    assert "5%-thick" in metadata["airfoil_adapter"]
    assert movement.num_steps == 169


def test_frozen_v4b_rejects_threshold_tuning_and_returns_finite_increment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = MANCINI_2017_CASES["fast_pitch"]
    t_star = np.linspace(0.0, case.observation_chords, 51)
    baseline = {
        "t_star": t_star,
        "pitch_deg": mancini_pitch_angle_deg(t_star, case),
        "CL": np.zeros_like(t_star),
        "CD": np.zeros_like(t_star),
    }
    with pytest.raises(ValueError, match="Lcrit=0.11"):
        apply_frozen_mancini_v4b(
            case,
            baseline,
            steps_per_chord=8,
            lesp_critical=np.nextafter(FROZEN_V4B_LESP_CRITICAL, np.inf),
        )
    result = apply_frozen_mancini_v4b(case, baseline, steps_per_chord=8)
    assert result["lesp_critical"] == FROZEN_V4B_LESP_CRITICAL
    assert result["steps_per_chord"] == 8
    assert np.isfinite(result["CL"]).all()
    assert np.isfinite(result["CD"]).all()
    assert np.max(np.abs(result["delta_CL"])) > 0.0
    assert "not fitted to Mancini" in result["lesp_provenance"]["source"]
    np.testing.assert_array_equal(result["CL"] - result["delta_CL"], baseline["CL"])

    monkeypatch.setattr(
        mancini_module,
        "MANCINI_FIG4_13B_CSV",
        mancini_module.MANCINI_FIG4_13B_CSV.with_name("must_not_be_read.csv"),
    )
    without_experiment = apply_frozen_mancini_v4b(case, baseline, steps_per_chord=8)
    np.testing.assert_array_equal(without_experiment["CL"], result["CL"])
    np.testing.assert_array_equal(without_experiment["CD"], result["CD"])
