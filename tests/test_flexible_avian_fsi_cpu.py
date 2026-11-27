"""CPU unit tests for verified bio-inspired planforms, articulated FSF wrist folding, and 3-DOF FSI."""

from __future__ import annotations

import math
import numpy as np
import pytest

from fluxvortex.cases.avian_aircraft import AVIAN_AIRCRAFT_REGISTRY
from fluxvortex.propulsion.flexible_avian_fsi import (
    FlexibleArticulatedWingKinematics,
    build_planform_distribution,
    build_tail_geometry_and_aero,
    build_wing_structural_model,
)


@pytest.mark.parametrize("aid", ["cloud_owl", "robofalcon_2", "npu_sparrow", "dove_2018", "dove_highspeed_2024"])
def test_planform_area_and_dimensions_exact(aid: str) -> None:
    spec = AVIAN_AIRCRAFT_REGISTRY[aid]
    y0, x_le, chord = build_planform_distribution(
        span_m=spec.span_m,
        area_m2=spec.area_m2,
        taper_ratio=spec.taper_ratio,
        quarter_chord_sweep_deg=spec.quarter_chord_sweep_deg,
        planform_kind=spec.planform_kind,
        wrist_eta=spec.fsf.wrist_eta if spec.fsf.enabled else 0.40,
        ns=spec.spanwise_panels,
    )
    area_num = float(getattr(np, "trapezoid", np.trapz)(chord, y0))
    assert math.isclose(area_num, spec.area_m2, rel_tol=1.0e-12)
    assert math.isclose(float(y0[-1] - y0[0]), spec.span_m, rel_tol=1.0e-12)
    assert np.all(chord > 0.0)
    # Symmetric port/starboard planform
    assert np.allclose(chord, chord[::-1], atol=1.0e-14)
    assert np.allclose(x_le, x_le[::-1], atol=1.0e-14)


def test_robofalcon_fsf_wrist_hinge_conserves_exact_spar_lengths() -> None:
    """RoboFalcon 2.0 articulated FSF wrist fold (-55 deg) & sweep (-20..+18 deg) must conserve
    exact inner & outer rigid LE-spar segment lengths to machine precision across the full cycle."""
    spec = AVIAN_AIRCRAFT_REGISTRY["robofalcon_2"]
    assert spec.fsf.enabled
    freq = 5.0
    kin = FlexibleArticulatedWingKinematics(
        span_m=spec.span_m,
        area_m2=spec.area_m2,
        alpha_deg=6.0,
        taper_ratio=spec.taper_ratio,
        quarter_chord_sweep_deg=spec.quarter_chord_sweep_deg,
        camber_zero_lift_deg=spec.camber_zero_lift_deg,
        planform_kind=spec.planform_kind,
        flap_amplitude_half_deg=spec.flap_amplitude_half_deg,
        flap_dihedral_deg=spec.flap_dihedral_deg,
        omega_rad_s=2.0 * math.pi * freq,
        start_ramp_s=0.0,
        fsf=spec.fsf,
        chordwise_panels=spec.chordwise_panels,
        spanwise_panels=spec.spanwise_panels,
        device="cpu",
    )
    ns = spec.spanwise_panels
    half = ns // 2
    j_wrist = half + int(round(kin._wrist_eta * half))

    # Reference undeformed segment lengths at neutral wrist (t where gamma_w=0 isn't needed; compare pairwise!)
    p_down = kin.positions_at(0.5 / freq, (0.0, 0.0, 0.0))[0].numpy()  # LE line at mid-downstroke
    p_up = kin.positions_at(0.0, (0.0, 0.0, 0.0))[0].numpy()           # LE line at mid-upstroke

    # Check exact inner & outer LE spar lengths conserved between downstroke & upstroke!
    l_inner_down = float(np.linalg.norm(p_down[j_wrist] - p_down[half]))
    l_inner_up = float(np.linalg.norm(p_up[j_wrist] - p_up[half]))
    l_outer_down = float(np.linalg.norm(p_down[-1] - p_down[j_wrist]))
    l_outer_up = float(np.linalg.norm(p_up[-1] - p_up[j_wrist]))
    assert math.isclose(l_inner_down, l_inner_up, rel_tol=1.0e-12)
    assert math.isclose(l_outer_down, l_outer_up, rel_tol=1.0e-12)

    # Check genuine articulated wrist bend angle between inner and outer spar vectors!
    # At mid-upstroke (t=0), gamma_w = -55 deg -> clear sharp wrist kink > 45 deg!
    v_in_up = p_up[j_wrist] - p_up[half]
    v_out_up = p_up[-1] - p_up[j_wrist]
    cos_bend_up = float(np.dot(v_in_up, v_out_up) / (np.linalg.norm(v_in_up) * np.linalg.norm(v_out_up)))
    bend_up_deg = math.degrees(math.acos(np.clip(cos_bend_up, -1.0, 1.0)))
    assert bend_up_deg > 50.0, f"Expected sharp upstroke wrist bend > 50 deg, got {bend_up_deg:.1f} deg"

    # At mid-downstroke (t=0.5/f), wing extends nearly straight (gamma_w=+8 deg, forward sweep -20 deg)
    span_down = float(p_down[-1, 1] - p_down[0, 1])
    span_up = float(p_up[-1, 1] - p_up[0, 1])
    assert span_down > 1.05
    assert span_up < 0.90


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
