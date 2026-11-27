"""CPU unit tests for ground-truth digitized planforms, RoboFalcon 2.0 Eq. S1 3-segment FSF, and FSI."""

from __future__ import annotations

import math
import numpy as np
import pytest

from fluxvortex.cases.avian_aircraft import AVIAN_AIRCRAFT_REGISTRY
from fluxvortex.cases.avian_wing_geometry_data import DIGITIZED_PLANFORMS, ROBOFALCON2_EQS1
from fluxvortex.propulsion.flexible_avian_fsi import (
    FlexibleArticulatedWingKinematics,
    build_planform_distribution,
    build_tail_geometry_and_aero,
    build_wing_structural_model,
)


@pytest.mark.parametrize("aid", ["cloud_owl", "robofalcon_2", "npu_sparrow", "dove_2018", "dove_highspeed_2024"])
def test_digitized_planform_area_and_dimensions_exact(aid: str) -> None:
    spec = AVIAN_AIRCRAFT_REGISTRY[aid]
    assert spec.planform_kind == "digitized"
    assert spec.planform_id == aid
    y0, x_le, chord = build_planform_distribution(
        span_m=spec.span_m,
        area_m2=spec.area_m2,
        taper_ratio=spec.taper_ratio,
        quarter_chord_sweep_deg=spec.quarter_chord_sweep_deg,
        planform_kind=spec.planform_kind,
        planform_id=spec.planform_id,
        wrist_eta=spec.fsf.wrist_eta if spec.fsf.enabled else 0.40,
        ns=spec.spanwise_panels,
    )
    area_num = float(getattr(np, "trapezoid", np.trapz)(chord, y0))
    assert math.isclose(area_num, spec.area_m2, rel_tol=1.0e-6)
    assert math.isclose(float(y0[-1] - y0[0]), spec.span_m, rel_tol=1.0e-12)
    assert np.all(chord > 0.0)
    assert np.allclose(chord, chord[::-1], atol=1.0e-14)
    assert np.allclose(x_le, x_le[::-1], atol=1.0e-14)


def test_digitized_planforms_match_published_station_chords() -> None:
    """Verify key published chord benchmarks (Cloud Owl Fig. 7, NPU-Sparrow Fig. 2, HS-Dove ICAS Table 1)."""

    # Cloud Owl: rectangular inboard ~290 mm out to s=560 mm (y=670 mm with 110 mm half-body)
    co = DIGITIZED_PLANFORMS["cloud_owl"]
    c_50 = float(np.interp(0.110 + 0.400, co.y_m, np.asarray(co.x_te_m) - np.asarray(co.x_le_m)))
    assert abs(c_50 - 0.289) < 0.003

    # High-Speed Dove ICAS Table 1: S-shaped chord widens to 125 mm at s=220 mm!
    hs = DIGITIZED_PLANFORMS["dove_highspeed_2024"]
    ch_hs = np.asarray(hs.x_te_m) - np.asarray(hs.x_le_m)
    for s_mm, c_mm in [(75.0, 100.0), (130.0, 100.0), (185.0, 120.0), (220.0, 125.0), (250.0, 115.0), (285.0, 85.0)]:
        c_val = float(np.interp(s_mm / 1000.0, hs.y_m, ch_hs)) * 1000.0
        assert abs(c_val - c_mm) < 0.5, f"HS-Dove s={s_mm}mm expected {c_mm}mm, got {c_val:.2f}mm"

    # Dove 2018: exact AR = 7.4 (b = 0.50 m -> S = 0.03378 m^2)
    dv = AVIAN_AIRCRAFT_REGISTRY["dove_2018"]
    assert abs(dv.aspect_ratio - 7.40) < 0.02


def test_robofalcon2_eqs1_in_plane_3segment_fsf_and_periodicity() -> None:
    """RoboFalcon 2.0 Eq. S1 3-segment FSF must conserve rigid Humerus/Radius/Manus lengths,
    reproduce Fig. 3J semispan contraction (600 mm -> 327 mm), fold IN-PLANE backward on upstroke,
    and be C^1 periodic at t/T = 0 vs 1."""

    spec = AVIAN_AIRCRAFT_REGISTRY["robofalcon_2"]
    assert spec.fsf.enabled and spec.fsf.fsf_mode == "robofalcon2_eqs1"
    freq = 5.0
    kin = FlexibleArticulatedWingKinematics(
        span_m=spec.span_m,
        area_m2=spec.area_m2,
        alpha_deg=0.0,
        taper_ratio=spec.taper_ratio,
        quarter_chord_sweep_deg=spec.quarter_chord_sweep_deg,
        camber_zero_lift_deg=spec.camber_zero_lift_deg,
        planform_kind=spec.planform_kind,
        planform_id=spec.planform_id,
        flap_amplitude_half_deg=spec.flap_amplitude_half_deg,
        flap_dihedral_deg=spec.flap_dihedral_deg,
        omega_rad_s=2.0 * math.pi * freq,
        start_ramp_s=0.0,
        fsf=spec.fsf,
        chordwise_panels=spec.chordwise_panels,
        spanwise_panels=spec.spanwise_panels,
        device="cpu",
    )

    # 1. Exact C^1 periodicity at t=0 vs t=1/freq
    p0, v0, _, _ = kin.positions_velocities_and_jacobians(0.0, np.zeros(3), np.zeros(3))
    p1, v1, _, _ = kin.positions_velocities_and_jacobians(1.0 / freq, np.zeros(3), np.zeros(3))
    assert float(np.max(np.abs(p1.numpy() - p0.numpy()))) < 1.0e-12
    assert float(np.max(np.abs(v1.numpy() - v0.numpy()))) < 1.0e-7

    # 2. Exact rigid bone length conservation across entire cycle (Humerus=110mm, Radius=180mm, Manus=325mm)
    semispans = []
    for phase in np.linspace(0.0, 1.0, 41):
        t = phase / freq
        sk = kin.robofalcon_skeleton_3d(t)
        assert sk is not None
        for s_idx in (0, 1):
            lh = float(np.linalg.norm(sk["humerus"][s_idx, 1] - sk["humerus"][s_idx, 0]))
            lr = float(np.linalg.norm(sk["radius"][s_idx, 1] - sk["radius"][s_idx, 0]))
            lm = float(np.linalg.norm(sk["manus"][s_idx, 1] - sk["manus"][s_idx, 0]))
            assert math.isclose(lh, ROBOFALCON2_EQS1.l_humerus_m, rel_tol=1.0e-12)
            assert math.isclose(lr, ROBOFALCON2_EQS1.l_radius_m, rel_tol=1.0e-12)
            assert math.isclose(lm, ROBOFALCON2_EQS1.l_manus_m, rel_tol=1.0e-12)
        st = kin.robofalcon2_eqs1_state(t)
        semispans.append(st["semispan_wing_plane_m"])

    # 3. Fig. 3J green semispan curve check: max ~600 mm (downstroke), min ~327 mm (mid-upstroke)
    assert 0.590 <= max(semispans) <= 0.605
    assert 0.315 <= min(semispans) <= 0.340

    # 4. In-plane aft folding check: at t/T = 0.51 (mid-upstroke), tip sweeps strongly AFT (+x > 160 mm)
    #    while at t/T = 0.25 (bottom reversal), tip sweeps strongly FORWARD (-x < -100 mm)!
    pos_bot = kin.positions_at(0.25 / freq)[0].numpy()
    pos_up = kin.positions_at(0.51 / freq)[0].numpy()
    assert pos_bot[-1, 0] < -0.10, f"Expected forward-swept tip at bottom reversal, got x={pos_bot[-1, 0]:.3f} m"
    assert pos_up[-1, 0] > 0.16, f"Expected in-plane aft-folded tip at mid-upstroke, got x={pos_up[-1, 0]:.3f} m"

    # 5. All UVLM panels remain strictly positive area & finite throughout full cycle
    for phase in (0.05, 0.25, 0.51, 0.78):
        fr = kin.evaluate(phase / freq)
        assert bool(np.all(fr.areas.numpy() > 1.0e-6))
        assert bool(np.all(np.isfinite(fr.normals_I.numpy())))


def test_flexible_structural_matrices_and_tail_aero() -> None:
    for aid in ("cloud_owl", "robofalcon_2", "npu_sparrow", "dove_2018", "dove_highspeed_2024"):
        spec = AVIAN_AIRCRAFT_REGISTRY[aid]
        assert spec.flexibility.enabled
        kin = FlexibleArticulatedWingKinematics(
            span_m=spec.span_m,
            area_m2=spec.area_m2,
            alpha_deg=5.0,
            taper_ratio=spec.taper_ratio,
            quarter_chord_sweep_deg=spec.quarter_chord_sweep_deg,
            camber_zero_lift_deg=spec.camber_zero_lift_deg,
            planform_kind=spec.planform_kind,
            planform_id=spec.planform_id,
            airfoil_camber_id=spec.airfoil_camber_id,
            airfoil_camber_scale=spec.airfoil_camber_scale,
            flap_amplitude_half_deg=spec.flap_amplitude_half_deg,
            flap_dihedral_deg=spec.flap_dihedral_deg,
            omega_rad_s=2.0 * math.pi * spec.nominal_frequency_hz,
            fsf=spec.fsf,
            chordwise_panels=spec.chordwise_panels,
            spanwise_panels=spec.spanwise_panels,
            device="cpu",
        )
        struct = build_wing_structural_model(kin, spec.flexibility)
        eig_m = np.linalg.eigvalsh(struct.mass)
        eig_k = np.linalg.eigvalsh(struct.k_linear)
        assert np.all(eig_m > 0.0)
        assert np.all(eig_k > 0.0)
        assert struct.k_camber_nl > 0.0

        tail = build_tail_geometry_and_aero(
            spec.tail, alpha_deg=6.0, airspeed_m_s=spec.cruise_speed_m_s, rho=spec.rho_kg_m3
        )
        assert tail["h_quads"].shape[0] > 0
        assert tail["drag_n"] > 0.0
