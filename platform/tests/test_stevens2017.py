from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
import pytest

from forward_flight_benchmarks.stevens2017 import (
    STEVENS_2017,
    STEVENS_EXPERIMENTAL_OBSERVABLES,
    STEVENS_EXPERIMENT_HAS_DRAG,
    STEVENS_FIG21_EXPERIMENT_CSV,
    build_stevens_movement,
    load_stevens_fig21_experiment,
    run_stevens_fluxv_ldvm_increment,
    stevens_pitch_angle_deg,
)


EXPECTED_CSV_SHA256 = "793c3c2af626de1f8e46e788e814f4735c903d3dfb198cd5e5257964fdac51fa"


@pytest.mark.parametrize("pivot_fraction", [0.0, 0.5])
def test_effective_ar4_geometry_area_and_pitch_pivot(pivot_fraction: float) -> None:
    case = STEVENS_2017
    assert case.span_m == pytest.approx(0.480)
    assert case.semispan_m == pytest.approx(0.240)
    assert case.area_m2 == pytest.approx(0.0576)
    assert case.span_m**2 / case.area_m2 == pytest.approx(4.0)

    movement, metadata = build_stevens_movement(pivot_fraction, quality="smoke")
    airplane_movement = movement.airplane_movements[0]
    airplane = airplane_movement.base_airplane
    wing = airplane.wings[0]
    wing_movement = airplane_movement.wing_movements[0]
    pivot_m = pivot_fraction * case.chord_m

    assert airplane.s_ref == pytest.approx(case.area_m2)
    assert airplane.c_ref == pytest.approx(case.chord_m)
    assert airplane.b_ref == pytest.approx(case.span_m)
    assert wing.symmetric is True
    assert wing.wing_cross_sections[-1].Lp_Wcsp_Lpp[1] == pytest.approx(case.semispan_m)
    assert wing.Ler_Gs_Cgs[0] == pytest.approx(-pivot_m)
    assert wing_movement.rotationPointOffset_Gs_Ler[0] == pytest.approx(pivot_m)
    assert (
        wing.Ler_Gs_Cgs[0] + wing_movement.rotationPointOffset_Gs_Ler[0]
    ) == pytest.approx(0.0)
    assert metadata["pivot_fraction_chord"] == pytest.approx(pivot_fraction)


def test_motion_preserves_reported_reduced_pitch_rate_and_smoothing() -> None:
    case = STEVENS_2017
    assert case.eldredge_smoothing == pytest.approx(11.0)
    assert case.reduced_pitch_rate == pytest.approx(np.pi / 8.0)
    assert case.reduced_pitch_rate == pytest.approx(0.392, abs=8.0e-4)
    assert case.pitch_rate_rad_s * case.chord_m / (
        2.0 * case.freestream_m_s
    ) == pytest.approx(case.reduced_pitch_rate)

    half_width = 1.0e-5
    angles = np.deg2rad(
        stevens_pitch_angle_deg(np.array([0.5 - half_width, 0.5 + half_width]))
    )
    pitch_rate_per_convected_chord = (angles[1] - angles[0]) / (2.0 * half_width)
    reconstructed_k = 0.5 * pitch_rate_per_convected_chord
    assert reconstructed_k == pytest.approx(np.pi / 8.0, rel=5.0e-5)
    assert stevens_pitch_angle_deg(np.array([0.5]))[0] == pytest.approx(22.5)


def test_fig21_csv_row_count_hash_grid_and_frozen_key_points() -> None:
    raw = STEVENS_FIG21_EXPERIMENT_CSV.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == EXPECTED_CSV_SHA256
    with STEVENS_FIG21_EXPERIMENT_CSV.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 501
    assert list(rows[0]) == [
        "s_over_c",
        "experiment_CL_leading_edge_axis",
        "experiment_CL_mid_chord_axis",
        "source_figure",
    ]
    assert {row["source_figure"] for row in rows} == {"21"}

    data = load_stevens_fig21_experiment()
    np.testing.assert_allclose(
        data["s_over_c"], np.linspace(0.0, 5.0, 501), rtol=0.0, atol=1.0e-15
    )
    frozen = {
        0: (-1.589376975, 0.022484912),
        2: (11.070177689, 0.509283781),
        5: (8.347666259, 1.239482084),
        100: (2.247277887, 3.153133411),
        200: (2.380956785, 2.473771390),
        500: (1.289856821, 1.609729336),
    }
    for index, (leading_edge, mid_chord) in frozen.items():
        assert data["CL_leading_edge_axis"][index] == pytest.approx(
            leading_edge, abs=5.0e-10
        )
        assert data["CL_mid_chord_axis"][index] == pytest.approx(mid_chord, abs=5.0e-10)


def test_experiment_contract_has_lift_but_no_drag_observation() -> None:
    assert STEVENS_EXPERIMENTAL_OBSERVABLES == ("CL",)
    assert STEVENS_EXPERIMENT_HAS_DRAG is False
    data = load_stevens_fig21_experiment()
    assert data["observed_channels"] == ("CL",)
    assert data["drag_available"] is False
    assert "CD" not in data
    assert "drag" not in data
    assert not any(
        channel.casefold() in {"cd", "drag"} for channel in data["observed_channels"]
    )

    with STEVENS_FIG21_EXPERIMENT_CSV.open(newline="", encoding="utf-8") as stream:
        fields = csv.DictReader(stream).fieldnames or []
    assert not any(
        "drag" in field.casefold() or "cd" in field.casefold() for field in fields
    )


def test_loader_rejects_changed_csv_hash(tmp_path: Path) -> None:
    changed = tmp_path / "changed.csv"
    changed.write_bytes(STEVENS_FIG21_EXPERIMENT_CSV.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_stevens_fig21_experiment(changed)


def test_v4_increment_exactly_reduces_to_uvlm_when_lesp_is_disabled() -> None:
    target = np.linspace(0.0, 5.0, 51)
    baseline = {
        "s_over_c": target,
        "pitch_deg": stevens_pitch_angle_deg(target),
        "CL": np.linspace(0.2, 0.8, target.size),
        "CD": np.linspace(0.01, 0.04, target.size),
    }
    corrected = run_stevens_fluxv_ldvm_increment(
        0.5,
        baseline,
        lesp_critical=1.0e6,
        threshold_source="unit-test disabled threshold",
        steps_per_chord=8,
    )
    np.testing.assert_array_equal(corrected["CL"], baseline["CL"])
    np.testing.assert_array_equal(corrected["CD"], baseline["CD"])
    np.testing.assert_array_equal(corrected["delta_CL"], np.zeros(target.size))
    np.testing.assert_array_equal(corrected["delta_CD"], np.zeros(target.size))
