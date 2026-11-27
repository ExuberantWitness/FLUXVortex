"""First-order electromechanical DC/BLDC motor + transmission/propeller power model.

Maps instantaneous aerodynamic hinge moments (flapping-wing ornithopters) or
trimmed thrust/drag loads (morphing fixed-wing drones) produced by FLUXVortex's
3D GPU UVLM solver into cycle-averaged electrical power consumption, loss
breakdowns, Cost of Transport (CoT), and battery endurance.

Motor model (first-order armature dynamics):
    L * di/dt + R * i + K_e * omega_m = V_cmd(t)
    tau_m(t) = K_t * (i(t) - i_0),   K_e = K_t = 60 / (2 * pi * KV)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FirstOrderMotorSpec:
    """First-order BLDC/DC motor, transmission, and electrical bus parameters."""

    name: str
    kv_rpm_per_v: float
    resistance_ohm: float
    no_load_current_a: float
    bus_voltage_v: float
    gear_ratio: float = 1.0
    gear_efficiency: float = 0.85
    esc_efficiency: float = 0.94
    inductance_h: float = 2.0e-5
    avionics_power_w: float = 1.0
    battery_capacity_wh: float | None = None
    battery_usable_fraction: float = 0.85
    propeller_diameter_m: float | None = None
    propeller_blade_efficiency: float = 0.78
    propeller_pitch_m: float | None = None
    regenerative_braking: bool = False
    mechanism_mode: str = "crank_rocker"  # "crank_rocker" | "direct_reciprocating"

    @property
    def kt_nm_per_a(self) -> float:
        return 60.0 / (2.0 * math.pi * self.kv_rpm_per_v)

    @property
    def ke_vs_per_rad(self) -> float:
        return self.kt_nm_per_a


@dataclass(frozen=True)
class ActuatorPowerBreakdown:
    """Cycle-averaged mechanical and electrical power breakdown."""

    aero_mech_power_w: float
    shaft_mech_power_w: float
    gearbox_loss_w: float
    copper_loss_w: float
    iron_loss_w: float
    esc_loss_w: float
    avionics_power_w: float
    total_elec_power_w: float
    mean_current_a: float
    peak_current_a: float
    mean_motor_rpm: float
    voltage_utilization: float
    motor_efficiency: float
    overall_efficiency: float
    cost_of_transport: float
    endurance_min: float | None
    peak_elec_power_w: float


def _solve_periodic_first_order_current(
    dt: float,
    inductance_h: float,
    resistance_ohm: float,
    back_emf_v: np.ndarray,
    v_cmd: np.ndarray,
) -> np.ndarray:
    """Solve periodic L * di/dt + R * i = v_cmd - back_emf via implicit backward Euler.

    Runs two periodic passes so initial transients decay completely for L/R << T.
    """

    n = len(v_cmd)
    if inductance_h <= 0.0:
        return (v_cmd - back_emf_v) / resistance_ohm
    alpha = inductance_h / dt
    denom = alpha + resistance_ohm
    i_curr = float((v_cmd[-1] - back_emf_v[-1]) / resistance_ohm)
    out = np.empty(n, dtype=np.float64)
    for _ in range(3):
        for k in range(n):
            i_curr = (alpha * i_curr + (v_cmd[k] - back_emf_v[k])) / denom
            out[k] = i_curr
    return out


def evaluate_flapping_motor_cycle(
    *,
    motor: FirstOrderMotorSpec,
    t_s: np.ndarray,
    flap_rate_rad_s: np.ndarray,
    aero_hinge_moment_nm: np.ndarray | None = None,
    aero_power_w: np.ndarray | None = None,
    frequency_hz: float,
    mass_kg: float,
    cruise_speed_m_s: float,
    wing_inertia_kg_m2: float = 0.0,
    flap_accel_rad_s2: np.ndarray | None = None,
    inertial_elastic_recovery: float = 0.70,
    gravity_m_s2: float = 9.81,
) -> ActuatorPowerBreakdown:
    """Evaluate 1st-order DC/BLDC motor electrical power over one converged flapping cycle.

    Parameters
    ----------
    t_s:
        Uniformly spaced timestamps over one complete flapping period [s], shape (K,).
    flap_rate_rad_s:
        Instantaneous wing flapping angular velocity d(phi)/dt [rad/s], shape (K,).
    aero_hinge_moment_nm:
        Instantaneous aerodynamic moment opposing flap_rate_rad_s summed over BOTH wings
        (positive when aerodynamic force opposes wing motion, so aero_hinge_moment * flap_rate
        is instantaneous aerodynamic power delivered by the actuator to the air), shape (K,).
    aero_power_w:
        Optional direct instantaneous aerodynamic power -sum(F_p . v_p) [W], shape (K,).
        When provided, takes precedence over aero_hinge_moment_nm * flap_rate_rad_s and
        captures flapping, twist, and span-folding aerodynamic work without dividing by zero
        at stroke reversals.
    wing_inertia_kg_m2:
        Combined flapping moment of inertia of both wings about their root hinges [kg*m^2].
    inertial_elastic_recovery:
        Fraction of inertial kinetic energy elastically/flywheel recovered across the stroke
        reversal (crank-rocker mechanisms + gearbox inertia naturally exchange kinetic energy
        over a cycle; 0.70 is typical for bird-scale crank-rocker ornithopters).
    """

    t_s = np.asarray(t_s, dtype=np.float64)
    phidot = np.asarray(flap_rate_rad_s, dtype=np.float64)
    if len(t_s) < 4:
        raise ValueError("need at least 4 samples per flapping cycle")
    dt = float((t_s[-1] - t_s[0]) / max(len(t_s) - 1, 1))
    if dt <= 0.0:
        dt = 1.0 / (frequency_hz * len(t_s))

    # Instantaneous aerodynamic mechanical power P_aero(t)
    if aero_power_w is not None:
        p_aero_inst = np.asarray(aero_power_w, dtype=np.float64)
    elif aero_hinge_moment_nm is not None:
        p_aero_inst = np.asarray(aero_hinge_moment_nm, dtype=np.float64) * phidot
    else:
        raise ValueError("must provide either aero_power_w or aero_hinge_moment_nm")
    if wing_inertia_kg_m2 > 0.0 and flap_accel_rad_s2 is not None:
        p_inert_inst = wing_inertia_kg_m2 * np.asarray(flap_accel_rad_s2, dtype=np.float64) * phidot
        # Unrecovered fraction of inertial acceleration power
        p_load_inst = p_aero_inst + (1.0 - inertial_elastic_recovery) * p_inert_inst
    else:
        p_load_inst = p_aero_inst

    if motor.regenerative_braking:
        p_pos_load = p_load_inst
    else:
        # Standard UAV ESCs cannot return negative braking power to the battery
        p_pos_load = np.maximum(p_load_inst, 0.0)

    eta_g = max(float(motor.gear_efficiency), 1.0e-6)
    p_shaft_inst = p_pos_load / eta_g

    if motor.mechanism_mode == "crank_rocker":
        # Continuous unidirectional rotation at crank speed 2*pi*f -> motor speed 2*pi*N*f
        omega_m = np.full_like(phidot, 2.0 * math.pi * motor.gear_ratio * frequency_hz)
        tau_m = p_shaft_inst / np.maximum(omega_m, 1.0e-9)
    elif motor.mechanism_mode == "direct_reciprocating":
        omega_m = np.abs(motor.gear_ratio * phidot)
        tau_m = np.where(omega_m > 1.0e-6, p_shaft_inst / np.maximum(omega_m, 1.0e-6), 0.0)
    else:
        raise ValueError(f"unknown mechanism_mode {motor.mechanism_mode!r}")

    kt = motor.kt_nm_per_a
    ke = motor.ke_vs_per_rad
    r = motor.resistance_ohm
    i0 = motor.no_load_current_a

    # Target current for required shaft torque + no-load current
    i_target = i0 + tau_m / kt
    di_dt = np.gradient(i_target, dt, edge_order=2)
    back_emf = ke * omega_m
    v_cmd = back_emf + r * i_target + motor.inductance_h * di_dt

    # Solve exact 1st-order electrical ODE for current response (preserves exact energy balance)
    i_act = _solve_periodic_first_order_current(dt, motor.inductance_h, r, back_emf, v_cmd)
    i_act = np.maximum(i_act, 0.0)

    p_motor_in_inst = np.maximum(v_cmd, 0.0) * i_act
    p_esc_in_inst = p_motor_in_inst / max(motor.esc_efficiency, 1.0e-6)
    p_total_inst = p_esc_in_inst + motor.avionics_power_w

    aero_mech_w = float(np.mean(p_aero_inst))
    shaft_mech_w = float(np.mean(p_shaft_inst))
    gearbox_loss_w = float(np.mean(p_shaft_inst - p_pos_load))
    copper_loss_w = float(np.mean((i_act ** 2) * r))
    iron_loss_w = float(np.mean(i0 * back_emf))
    p_motor_in_w = float(np.mean(p_motor_in_inst))
    esc_loss_w = float(np.mean(p_esc_in_inst - p_motor_in_inst))
    total_elec_w = float(np.mean(p_total_inst))
    peak_elec_w = float(np.max(p_total_inst))

    mean_rpm = float(np.mean(omega_m) * 60.0 / (2.0 * math.pi))
    v_util = float(np.max(np.abs(v_cmd)) / max(motor.bus_voltage_v, 1.0e-9))
    motor_eff = shaft_mech_w / max(p_motor_in_w, 1.0e-9)
    overall_eff = max(aero_mech_w, 0.0) / max(total_elec_w, 1.0e-9)
    cot = total_elec_w / max(mass_kg * gravity_m_s2 * max(cruise_speed_m_s, 1.0e-6), 1.0e-9)
    endurance_min = (
        60.0 * motor.battery_capacity_wh * motor.battery_usable_fraction / max(total_elec_w, 1.0e-9)
        if motor.battery_capacity_wh is not None
        else None
    )
    return ActuatorPowerBreakdown(
        aero_mech_power_w=aero_mech_w,
        shaft_mech_power_w=shaft_mech_w,
        gearbox_loss_w=gearbox_loss_w,
        copper_loss_w=copper_loss_w,
        iron_loss_w=iron_loss_w,
        esc_loss_w=esc_loss_w,
        avionics_power_w=motor.avionics_power_w,
        total_elec_power_w=total_elec_w,
        mean_current_a=float(np.mean(i_act)),
        peak_current_a=float(np.max(i_act)),
        mean_motor_rpm=mean_rpm,
        voltage_utilization=v_util,
        motor_efficiency=motor_eff,
        overall_efficiency=overall_eff,
        cost_of_transport=cot,
        endurance_min=endurance_min,
        peak_elec_power_w=peak_elec_w,
    )


def evaluate_propeller_motor_cruise(
    *,
    motor: FirstOrderMotorSpec,
    thrust_n: float,
    cruise_speed_m_s: float,
    mass_kg: float,
    rho_kg_m3: float = 1.225,
    gravity_m_s2: float = 9.81,
) -> ActuatorPowerBreakdown:
    """Evaluate 1st-order DC/BLDC motor + propeller cruise electrical power.

    Uses Froude actuator-disk momentum theory paired with propeller blade profile
    efficiency and the steady-state 1st-order BLDC motor equations.
    """

    thrust_n = max(float(thrust_n), 0.0)
    u = max(float(cruise_speed_m_s), 1.0e-3)
    p_aero_w = thrust_n * u

    d_prop = float(motor.propeller_diameter_m or 0.2032)  # default 8-inch prop
    area_prop = 0.25 * math.pi * d_prop * d_prop
    # Froude ideal propulsive efficiency: eta_ideal = 2 / (1 + sqrt(1 + T / (0.5 rho A_p U^2)))
    ct_disk = thrust_n / max(0.5 * rho_kg_m3 * area_prop * u * u, 1.0e-9)
    eta_ideal = 2.0 / (1.0 + math.sqrt(1.0 + ct_disk))
    eta_prop = eta_ideal * float(motor.propeller_blade_efficiency)
    p_shaft_w = p_aero_w / max(eta_prop, 1.0e-6)

    # Estimate cruise propeller angular speed from pitch (or advance ratio J ~= 0.55)
    pitch_m = float(motor.propeller_pitch_m or (0.5 * d_prop))
    # Effective pitch advance including slip (~15% slip in cruise)
    n_rps = u / max(0.82 * pitch_m, 1.0e-3)
    omega_prop = 2.0 * math.pi * n_rps
    omega_m = omega_prop * motor.gear_ratio
    eta_g = max(float(motor.gear_efficiency), 1.0e-6)
    p_motor_shaft_w = p_shaft_w / eta_g
    tau_m = p_motor_shaft_w / max(omega_m, 1.0e-9)

    kt = motor.kt_nm_per_a
    ke = motor.ke_vs_per_rad
    r = motor.resistance_ohm
    i0 = motor.no_load_current_a

    i_ss = i0 + tau_m / kt
    back_emf = ke * omega_m
    v_ss = back_emf + r * i_ss
    p_motor_in_w = v_ss * i_ss
    p_esc_in_w = p_motor_in_w / max(motor.esc_efficiency, 1.0e-6)
    total_elec_w = p_esc_in_w + motor.avionics_power_w

    copper_loss_w = (i_ss ** 2) * r
    iron_loss_w = i0 * back_emf
    esc_loss_w = p_esc_in_w - p_motor_in_w
    gearbox_loss_w = p_motor_shaft_w - p_shaft_w

    cot = total_elec_w / max(mass_kg * gravity_m_s2 * u, 1.0e-9)
    endurance_min = (
        60.0 * motor.battery_capacity_wh * motor.battery_usable_fraction / max(total_elec_w, 1.0e-9)
        if motor.battery_capacity_wh is not None
        else None
    )
    return ActuatorPowerBreakdown(
        aero_mech_power_w=p_aero_w,
        shaft_mech_power_w=p_motor_shaft_w,
        gearbox_loss_w=gearbox_loss_w,
        copper_loss_w=copper_loss_w,
        iron_loss_w=iron_loss_w,
        esc_loss_w=esc_loss_w,
        avionics_power_w=motor.avionics_power_w,
        total_elec_power_w=total_elec_w,
        mean_current_a=i_ss,
        peak_current_a=i_ss,
        mean_motor_rpm=omega_m * 60.0 / (2.0 * math.pi),
        voltage_utilization=v_ss / max(motor.bus_voltage_v, 1.0e-9),
        motor_efficiency=p_motor_shaft_w / max(p_motor_in_w, 1.0e-9),
        overall_efficiency=p_aero_w / max(total_elec_w, 1.0e-9),
        cost_of_transport=cot,
        endurance_min=endurance_min,
        peak_elec_power_w=total_elec_w,
    )
