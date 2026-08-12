from __future__ import annotations

import numpy as np

from forward_flight_benchmarks.cases import (
    IZRAELEVITZ_2017,
    IZRAELEVITZ_2017_FIG11,
    IZRAELEVITZ_2017_FIG14_SCHERER,
    YANG_2023,
    YANG_2025,
    fourbar_extrema_deg,
    fourbar_normalized_spacing,
    izraelevitz_euler_spacing,
    yang_fourbar_spacing,
)
from forward_flight_benchmarks.ptera_adapter import (
    build_izraelevitz_movement,
    build_izraelevitz_fig11_movement,
    build_izraelevitz_scherer_movement,
    build_yang2025_movement,
    build_yang_movement,
    run_model,
)
from forward_flight_benchmarks.yang2025_robofalcon import (
    run_yang2025_robofalcon_transfer,
)


def test_izraelevitz_elliptic_geometry_invariants() -> None:
    case = IZRAELEVITZ_2017
    assert np.isclose(case.span_m, np.pi * 6.0 * 0.1 / 4.0)
    assert np.isclose(case.area_m2, np.pi * case.span_m * 0.1 / 4.0)
    assert np.isclose(case.span_m**2 / case.area_m2, 6.0)
    assert np.isclose(case.chord_m(0.0), 0.1)
    assert np.isclose(case.chord_m(case.semispan_m), 0.0)
    fig11 = IZRAELEVITZ_2017_FIG11
    assert np.isclose(fig11.span_m**2 / fig11.area_m2, 3.0)
    assert np.isclose(fig11.reduced_frequency_midspan, 0.2 * np.pi)
    assert np.isclose(fig11.pitch_amplitude_deg, -28.3038073, atol=1.0e-6)


def test_izraelevitz_scherer_experiment_geometry_and_motion() -> None:
    case = IZRAELEVITZ_2017_FIG14_SCHERER
    assert np.isclose(case.chord_m, 4.0 * 0.0254)
    assert np.isclose(case.span_m, 12.0 * 0.0254)
    assert np.isclose(case.span_m**2 / case.area_m2, 3.0)
    assert np.isclose(case.freestream_m_s / (case.frequency_hz * case.chord_m), 6.0)
    assert np.isclose(case.reduced_frequency_midspan, np.pi / 6.0)
    assert np.isclose(case.strouhal, 0.2)
    assert case.profile_drag_coefficient == 0.057
    assert case.scherer_static_profile_drag_coefficient == 0.027

    movement, metadata = build_izraelevitz_scherer_movement(15.0, 60.0, "smoke")
    assert movement.num_steps > 64
    assert metadata["grid_chord_semispan"] == [2, 6]
    assert metadata["pivot_fraction_chord"] == 0.75
    assert metadata["heave_law"] == "z=h*cos(omega*t)"
    assert metadata["pitch_law"] == "theta=theta_max*cos(omega*t+psi)"


def test_yang_fourbar_recovers_published_extrema() -> None:
    lower, upper = fourbar_extrema_deg(YANG_2023)
    assert abs(lower - YANG_2023.target_downstroke_deg) < 0.02
    assert abs(upper - YANG_2023.target_upstroke_deg) < 0.02
    phase = np.linspace(0.0, 2.0 * np.pi, 201)
    spacing = yang_fourbar_spacing(phase)
    assert abs(spacing[0]) < 1.0e-10
    assert abs(spacing[-1]) < 1.0e-10
    assert np.isclose(0.5 * (spacing.max() - spacing.min()), 1.0, atol=0.01)


def test_yang2025_formal_case_is_distinct_and_auditable() -> None:
    case = YANG_2025
    lower, upper = fourbar_extrema_deg(case)
    assert case.phi0_deg == -14.5
    assert case.mechanism_phi0_deg == 14.5
    # Rounded published links approximately recover the stated -30/+40 range;
    # the exact paper run instead used an unpublished LDS motion history.
    assert abs(lower - case.target_downstroke_deg) < 3.0
    assert abs(upper - case.target_upstroke_deg) < 1.0
    assert np.isclose(case.initial_core_radius_m, 1.3e-6)
    assert np.isclose(case.reynolds, 48639.45578231293)
    phase = np.linspace(0.0, 2.0 * np.pi, 201)
    spacing = fourbar_normalized_spacing(phase, case)
    assert np.isclose(spacing[0], 0.0, atol=1.0e-10)
    assert np.isclose(spacing[-1], 0.0, atol=1.0e-10)
    assert np.isclose(0.5 * (spacing.max() - spacing.min()), 1.0, atol=0.01)


def test_izraelevitz_euler_custom_waveforms_are_valid() -> None:
    phase = np.linspace(0.0, 2.0 * np.pi, 201)
    for component in range(3):
        amplitude, spacing = izraelevitz_euler_spacing(component)
        values = spacing(phase)
        assert amplitude > 0.0
        assert abs(values[0]) < 1.0e-10
        assert abs(values[-1]) < 1.0e-10
        assert np.isclose(0.5 * (values.max() - values.min()), 1.0, atol=0.01)


def test_ptera_movement_builders_smoke() -> None:
    yang, yang_meta = build_yang_movement(10.0, "smoke")
    yang2025, yang2025_meta = build_yang2025_movement(10.0, "smoke")
    izra, izra_meta = build_izraelevitz_movement("smoke")
    fig11, fig11_meta = build_izraelevitz_fig11_movement("smoke")
    assert yang.num_steps > 20
    assert yang2025.num_steps > 20
    assert izra.num_steps > 24
    assert fig11.num_steps > 24
    assert yang_meta["grid_chord_span"] == [2, 4]
    assert yang2025_meta["grid_chord_span"] == [2, 4]
    assert yang2025_meta["wing_root_offset_m"] == YANG_2025.wing_root_offset_m
    assert yang2025_meta["kinematics"].startswith("nominal-fourbar")
    assert izra_meta["grid_chord_semispan"] == [2, 6]
    assert fig11_meta["grid_chord_semispan"] == [2, 7]


def test_yang2025_robofalcon_transfer_is_explicit_and_finite() -> None:
    result = run_yang2025_robofalcon_transfer(
        10.0, settings=(6, 24, 1), output_samples=24
    )
    assert result["phase"].shape == (24,)
    assert np.allclose(result["drag_n"], -result["thrust_n"])
    assert np.isfinite(result["mean_lift_n"])
    assert np.isfinite(result["mean_drag_n"])
    assert result["metadata"]["cross_domain_transfer"] is True
    assert "not native" in result["model_semantics"]


def test_yang2025_last_cycle_excludes_repeated_endpoint() -> None:
    movement, _ = build_yang2025_movement(15.0, "smoke")
    result = run_model(
        movement,
        "ptera_prescribed_wake_uvlm",
        period_s=YANG_2025.period_s,
        rho=YANG_2025.rho_kg_m3,
        speed=YANG_2025.freestream_m_s,
        area=YANG_2025.area_m2,
        output_samples=20,
    )
    assert result["source_cycle_sample_count"] == 20
    assert result["source_cycle_step_range"] == [20, 39]
