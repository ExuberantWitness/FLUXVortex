"""RoboFalcon 2.0 coefficient-transfer adapter for the Yang 2025 rigid wing.

This is deliberately not the native RoboFalcon/RoboEagle validation case.
It transfers the published RoboFalcon S3--S4 coefficient law to Yang's
rectangular single-wing geometry and reconstructed four-bar kinematics.  The
distinction is returned in the model semantics and must remain in plot labels.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from robofalcon2_aero import force_coefficients, unsteady_factor

from .cases import YANG_2025, Yang2025RigidCase, fourbar_zero_phase_rad


MODEL_KEY = "robofalcon2_coefficient_transfer"
MODEL_SEMANTICS = (
    "RoboFalcon 2.0 Eqs. S3-S4 coefficient transfer to the Yang 2025 "
    "rectangular single wing and nominal four-bar motion. The coefficients "
    "were calibrated on the RoboFalcon/RoboEagle wing, not on this geometry "
    "or Reynolds number; this is a cross-domain diagnostic, not native "
    "RoboFalcon validation."
)


def _rotation(pitch_rad: float, flap_rad: np.ndarray) -> np.ndarray:
    """Return active ``Rx(flap) @ Ry(pitch)`` matrices."""

    cp, sp = np.cos(pitch_rad), np.sin(pitch_rad)
    cf, sf = np.cos(flap_rad), np.sin(flap_rad)
    matrices = np.zeros(flap_rad.shape + (3, 3), dtype=float)
    matrices[..., 0, 0] = cp
    matrices[..., 0, 2] = sp
    matrices[..., 1, 0] = sf * sp
    matrices[..., 1, 1] = cf
    matrices[..., 1, 2] = -sf * cp
    matrices[..., 2, 0] = -cf * sp
    matrices[..., 2, 1] = sf
    matrices[..., 2, 2] = cf * cp
    return matrices


def _normalise(vectors: np.ndarray, name: str) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    if np.any(norms <= 1.0e-12):
        raise ValueError(f"degenerate {name} direction")
    return vectors / norms


def run_yang2025_robofalcon_transfer(
    aoa_deg: float,
    *,
    quality: str = "full",
    case: Yang2025RigidCase = YANG_2025,
    settings: tuple[int, int, int] | None = None,
    output_samples: int = 128,
) -> dict[str, Any]:
    """Run the S3--S4 strip law on one Yang installation-angle condition."""

    start = time.perf_counter()
    if settings is None:
        ns, steps_per_cycle, cycles = (
            (12, 100, 2) if quality == "smoke" else (48, 400, 3)
        )
    else:
        ns, steps_per_cycle, cycles = settings
    if min(ns, steps_per_cycle, cycles, output_samples) <= 0:
        raise ValueError("all discretization and output settings must be positive")

    period_s = case.period_s
    omega = 2.0 * np.pi / period_s
    total_steps = steps_per_cycle * cycles
    time_s = np.arange(total_steps, dtype=float) * period_s / steps_per_cycle
    crank_phase = omega * time_s + fourbar_zero_phase_rad(case)

    # Evaluate the published four-bar directly, including its non-sinusoidal
    # timing.  A small centered derivative provides angular rate without
    # imposing a harmonic approximation.
    from .cases import fourbar_flap_angle_deg

    flap_rad = np.deg2rad(fourbar_flap_angle_deg(crank_phase, case))
    derivative_dt = period_s * 1.0e-6
    flap_plus = np.deg2rad(
        fourbar_flap_angle_deg(crank_phase + omega * derivative_dt, case)
    )
    flap_minus = np.deg2rad(
        fourbar_flap_angle_deg(crank_phase - omega * derivative_dt, case)
    )
    flap_rate = (flap_plus - flap_minus) / (2.0 * derivative_dt)

    pitch_rad = np.deg2rad(float(aoa_deg))
    rotations = _rotation(pitch_rad, flap_rad)
    chord_hat = rotations @ np.array([1.0, 0.0, 0.0])
    span_hat = rotations @ np.array([0.0, 1.0, 0.0])
    chord_hat = _normalise(chord_hat, "chord")
    span_hat = _normalise(span_hat, "span")
    normal_hat = _normalise(np.cross(chord_hat, span_hat), "normal")

    dy = case.span_m / ns
    radius = case.wing_root_offset_m + (np.arange(ns, dtype=float) + 0.5) * dy
    # The leading-edge reference point lies on the rigid span ray.  Its speed
    # under rotation about the body-x flapping joint is omega_flap x r.
    local_points = np.zeros((ns, 3), dtype=float)
    local_points[:, 1] = radius
    points = np.einsum("tij,sj->tsi", rotations, local_points)
    angular_velocity = np.zeros((total_steps, 3), dtype=float)
    angular_velocity[:, 0] = flap_rate
    reference_velocity = np.cross(angular_velocity[:, None, :], points)

    air_velocity = np.array([case.freestream_m_s, 0.0, 0.0], dtype=float)
    relative_velocity = air_velocity[None, None, :] - reference_velocity
    span_3d = span_hat[:, None, :]
    chord_3d = chord_hat[:, None, :]
    normal_3d = normal_hat[:, None, :]
    section_velocity = relative_velocity - (
        np.sum(relative_velocity * span_3d, axis=-1, keepdims=True) * span_3d
    )
    local_speed = np.linalg.norm(section_velocity, axis=-1)
    if np.any(local_speed <= 1.0e-10):
        raise ValueError("local section speed reached zero")
    flow_hat = section_velocity / local_speed[..., None]
    alpha_local = np.arctan2(
        np.sum(section_velocity * normal_3d, axis=-1),
        np.sum(section_velocity * chord_3d, axis=-1),
    )
    cl, cd, _ = force_coefficients(alpha_local)
    reference_speed = np.linalg.norm(reference_velocity, axis=-1)
    body_x_speed = np.abs(relative_velocity[..., 0])
    cus = unsteady_factor(reference_speed, body_x_speed)

    dynamic_factor = 0.5 * case.rho_kg_m3 * local_speed**2 * cus
    strip_area = case.chord_m * dy
    lift_magnitude = dynamic_factor * cl * strip_area
    drag_magnitude = dynamic_factor * cd * strip_area
    lift_hat = _normalise(np.cross(flow_hat, span_3d), "lift")
    strip_force = (
        lift_magnitude[..., None] * lift_hat
        + drag_magnitude[..., None] * flow_hat
    )
    # Yang's validation article reports one physical wing, so no left/right
    # factor is applied here.
    force = np.sum(strip_force, axis=1)
    lift_n = force[:, 2]
    drag_n = force[:, 0]

    keep = time_s >= (time_s[-1] - period_s + period_s / steps_per_cycle)
    phase = np.mod(time_s[keep] / period_s, 1.0)
    order = np.argsort(phase)
    target = np.arange(output_samples, dtype=float) / output_samples
    lift = np.interp(target, phase[order], lift_n[keep][order], period=1.0)
    drag = np.interp(target, phase[order], drag_n[keep][order], period=1.0)
    thrust = -drag
    q_area = 0.5 * case.rho_kg_m3 * case.freestream_m_s**2 * case.area_m2

    result: dict[str, Any] = {
        "phase": target,
        "lift_n": lift,
        "drag_n": drag,
        "thrust_n": thrust,
        "CL": lift / q_area,
        "CD": drag / q_area,
        "CT": thrust / q_area,
        "mean_lift_n": float(np.mean(lift)),
        "mean_drag_n": float(np.mean(drag)),
        "mean_thrust_n": float(np.mean(thrust)),
        "mean_CL": float(np.mean(lift / q_area)),
        "mean_CD": float(np.mean(drag / q_area)),
        "mean_CT": float(np.mean(thrust / q_area)),
        "runtime_s": time.perf_counter() - start,
        "particle_count": 0,
        "model_semantics": MODEL_SEMANTICS,
        "metadata": {
            "model": MODEL_KEY,
            "adapter": "Yang geometry + nominal four-bar; RoboFalcon S3-S4 coefficients",
            "coefficient_calibration_domain": "RoboFalcon/RoboEagle wing, 6-12 m/s",
            "cross_domain_transfer": True,
            "strips": ns,
            "steps_per_cycle": steps_per_cycle,
            "cycles": cycles,
            "aoa_deg": float(aoa_deg),
            "wing_root_offset_m": case.wing_root_offset_m,
            "single_wing_area_m2": case.area_m2,
            "cus_applied_to": ["CL", "CD"],
            "freestream_definition": "fixed [5.5, 0, 0] m/s; AoA rotates wing",
            "alpha_local_deg_min": float(np.rad2deg(np.min(alpha_local))),
            "alpha_local_deg_max": float(np.rad2deg(np.max(alpha_local))),
            "cus_min": float(np.min(cus)),
            "cus_max": float(np.max(cus)),
        },
    }
    arrays = [lift, drag, thrust, result["CL"], result["CD"], result["CT"]]
    if not all(np.all(np.isfinite(values)) for values in arrays):
        raise FloatingPointError("RoboFalcon coefficient-transfer result is non-finite")
    return result


__all__ = [
    "MODEL_KEY",
    "MODEL_SEMANTICS",
    "run_yang2025_robofalcon_transfer",
]
