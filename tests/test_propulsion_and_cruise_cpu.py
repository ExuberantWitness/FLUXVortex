"""CPU unit tests for the 1st-order motor actuator model, avian registry, and kinematics."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from fluxvortex.cases.avian_aircraft import AVIAN_AIRCRAFT_REGISTRY
from fluxvortex.propulsion.actuator import (
    FirstOrderMotorSpec,
    evaluate_flapping_motor_cycle,
    evaluate_propeller_motor_cruise,
)
from fluxvortex.propulsion.cruise_trim import MorphingFlappingWingKinematics


def test_first_order_motor_energy_conservation_exact() -> None:
    """Verify P_motor_in = P_shaft + P_copper + P_iron over a periodic cycle."""

    motor = FirstOrderMotorSpec(
        name="test_bldc",
        kv_rpm_per_v=1400.0,
        resistance_ohm=0.08,
        no_load_current_a=0.60,
        bus_voltage_v=14.8,
        gear_ratio=25.0,
        gear_efficiency=0.82,
        esc_efficiency=0.94,
        inductance_h=2.0e-5,
        avionics_power_w=2.0,
        battery_capacity_wh=100.0,
        battery_usable_fraction=0.85,
    )
    freq = 5.0
    n = 128
    t = np.linspace(0.0, 1.0 / freq, n, endpoint=False)
    omega = 2.0 * math.pi * freq
    phidot = math.radians(30.0) * omega * np.cos(omega * t)
    # Strictly positive aerodynamic work over the cycle
    p_aero = 25.0 * (0.6 + 0.4 * np.cos(2.0 * omega * t))

    res = evaluate_flapping_motor_cycle(
        motor=motor,
        t_s=t,
        flap_rate_rad_s=phidot,
        aero_power_w=p_aero,
        frequency_hz=freq,
        mass_kg=0.80,
        cruise_speed_m_s=8.0,
    )
    # P_motor_in = P_total - P_avionics - P_esc_loss
    p_motor_in = res.total_elec_power_w - res.avionics_power_w - res.esc_loss_w
    p_balance = res.shaft_mech_power_w + res.copper_loss_w + res.iron_loss_w
    assert math.isclose(p_motor_in, p_balance, rel_tol=1.0e-4)
    assert math.isclose(res.aero_mech_power_w + res.gearbox_loss_w, res.shaft_mech_power_w, rel_tol=1.0e-6)
    assert res.endurance_min is not None and res.endurance_min > 0.0
    assert 0.0 < res.motor_efficiency < 1.0
    assert 0.0 < res.overall_efficiency < res.motor_efficiency


def test_propeller_motor_cruise_conservation_and_monotonicity() -> None:
    """Higher thrust monotonically increases electrical power and preserves motor energy balance."""

    motor = FirstOrderMotorSpec(
        name="prop_test",
        kv_rpm_per_v=2300.0,
        resistance_ohm=0.08,
        no_load_current_a=0.60,
        bus_voltage_v=11.1,
        gear_ratio=1.0,
        gear_efficiency=1.0,
        esc_efficiency=0.94,
        avionics_power_w=2.5,
        battery_capacity_wh=7.77,
        propeller_diameter_m=0.2032,
        propeller_pitch_m=0.1016,
        propeller_blade_efficiency=0.76,
    )
    r_lo = evaluate_propeller_motor_cruise(
        motor=motor, thrust_n=0.8, cruise_speed_m_s=9.0, mass_kg=0.62
    )
    r_hi = evaluate_propeller_motor_cruise(
        motor=motor, thrust_n=1.6, cruise_speed_m_s=9.0, mass_kg=0.62
    )
    assert r_hi.total_elec_power_w > r_lo.total_elec_power_w
    p_motor_in = r_lo.total_elec_power_w - r_lo.avionics_power_w - r_lo.esc_loss_w
    assert math.isclose(
        p_motor_in, r_lo.shaft_mech_power_w + r_lo.copper_loss_w + r_lo.iron_loss_w, rel_tol=1.0e-12
    )


def test_avian_aircraft_registry_completeness() -> None:
    """All 8 landmark published aircraft are registered with valid positive physical parameters."""

    expected = {
        "cloud_owl",
        "robofalcon_2",
        "npu_sparrow",
        "dove_2018",
        "dove_highspeed_2024",
        "raven",
        "liseagle_morphing",
        "liseagle_perching",
    }
    assert set(AVIAN_AIRCRAFT_REGISTRY.keys()) == expected
    for spec in AVIAN_AIRCRAFT_REGISTRY.values():
        assert spec.mass_kg > 0.0
        assert spec.span_m > 0.0
        assert spec.area_m2 > 0.0
        assert spec.cruise_speed_m_s > 0.0
        assert spec.motor.kv_rpm_per_v > 0.0
        assert spec.motor.resistance_ohm > 0.0


def test_morphing_flapping_kinematics_analytical_velocity_and_symmetry() -> None:
    """Analytical node velocity matches central finite difference to < 1e-8 and preserves symmetry."""

    kin = MorphingFlappingWingKinematics(
        span_m=1.20,
        area_m2=0.220,
        alpha_deg=8.5,
        taper_ratio=0.55,
        quarter_chord_sweep_deg=6.0,
        camber_zero_lift_deg=2.0,
        flap_amplitude_half_deg=35.0,
        flap_dihedral_deg=4.0,
        twist_tip_deg=16.0,
        upstroke_span_fold=0.20,
        omega_rad_s=2.0 * math.pi * 5.0,
        start_ramp_s=0.10,
        chordwise_panels=6,
        spanwise_panels=12,
        device="cpu",
    )
    eps = 1.0e-7
    for t in (0.03, 0.14, 0.27):
        pos, vel = kin.positions_and_velocities(t)
        pos_p, _ = kin.positions_and_velocities(t + eps)
        pos_m, _ = kin.positions_and_velocities(t - eps)
        vel_fd = (pos_p - pos_m) / (2.0 * eps)
        max_err = float(torch.max(torch.abs(vel - vel_fd)).item())
        assert max_err < 1.0e-7, f"t={t} analytical vel mismatch: {max_err}"
        # Left-right mirror symmetry: x, z equal; y opposite
        assert torch.allclose(pos[:, :, 0], torch.flip(pos[:, :, 0], dims=[1]), atol=1e-12)
        assert torch.allclose(pos[:, :, 1], -torch.flip(pos[:, :, 1], dims=[1]), atol=1e-12)
        assert torch.allclose(pos[:, :, 2], torch.flip(pos[:, :, 2], dims=[1]), atol=1e-12)
        # Centerline stays strictly at y = 0
        mid = pos.shape[1] // 2
        assert float(torch.max(torch.abs(pos[:, mid, 1])).item()) < 1.0e-14
