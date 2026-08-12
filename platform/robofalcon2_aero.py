"""Independent reproduction of the RoboFalcon 2.0 strip-theory aero model.

This module implements equations (3)--(5) in Chen et al., Science Advances
11, eadx0465 (2025), and equations (S2)--(S4) in its Supplementary Materials.
It deliberately does not import or call the FLUXV UVLM/LESP/L-B solver.

The published model leaves the strip count and the explicit construction of
``alpha_loc``, ``v_loc`` and ``v_ref`` to the MuJoCo geometry.  The adapter in
``run_robofalcon2_184.py`` supplies the frozen RoboEagle geometry and
flap/twist kinematics associated with the 184-condition benchmark.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

import _v2_robogeom as robogeom


PAPER_DOI = "10.1126/sciadv.adx0465"
DEFAULT_RHO_KG_M3 = 1.225
DEFAULT_HALF_SPAN_M = 0.80
DEFAULT_FLAP_AMPLITUDE_DEG = 22.5
DEFAULT_TWIST_PHASE_DEG = 90.0


@dataclass(frozen=True)
class RoboFalcon2Case:
    """One benchmark condition expressed in the published strip model."""

    airspeed_m_s: float
    frequency_hz: float
    nominal_twist_deg: float
    aoa_deg: float
    ns: int = 16
    steps_per_cycle: int = 400
    n_cycle: int = 5
    rho_kg_m3: float = DEFAULT_RHO_KG_M3
    half_span_m: float = DEFAULT_HALF_SPAN_M
    flap_amplitude_deg: float = DEFAULT_FLAP_AMPLITUDE_DEG
    twist_phase_deg: float = DEFAULT_TWIST_PHASE_DEG

    def validate(self) -> None:
        finite_values = (
            self.airspeed_m_s,
            self.frequency_hz,
            self.nominal_twist_deg,
            self.aoa_deg,
            self.rho_kg_m3,
            self.half_span_m,
            self.flap_amplitude_deg,
            self.twist_phase_deg,
        )
        if not all(np.isfinite(value) for value in finite_values):
            raise ValueError("case contains a non-finite scalar")
        if self.airspeed_m_s < 0.0:
            raise ValueError("airspeed_m_s must be non-negative")
        if self.frequency_hz <= 0.0:
            raise ValueError("frequency_hz must be positive")
        if self.ns < 1:
            raise ValueError("ns must be positive")
        if self.steps_per_cycle < 8:
            raise ValueError("steps_per_cycle must be at least 8")
        if self.n_cycle < 1:
            raise ValueError("n_cycle must be positive")
        if self.rho_kg_m3 <= 0.0:
            raise ValueError("rho_kg_m3 must be positive")
        if self.half_span_m <= 0.0:
            raise ValueError("half_span_m must be positive")


def force_coefficients(alpha_loc_rad: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the published Eq. (S4) ``CL``, ``CD`` and ``CM``.

    ``alpha_loc`` and all fitted phase offsets are radians.  The PDF calls the
    equations Fourier fits but omits ``sin`` before the third CL harmonic;
    ``-0.1547*sin(3*alpha_loc + 1.627)`` is the unique Fourier-consistent
    reading and is recorded as an explicit typesetting repair in the run
    manifest.  All signs and phase offsets below follow the PDF image itself.
    """

    alpha = np.asarray(alpha_loc_rad, dtype=float)
    cl = (
        0.8172
        + 0.4545 * np.sin(alpha + 0.5146)
        + 0.4109 * np.sin(2.0 * alpha + 0.2233)
        - 0.1547 * np.sin(3.0 * alpha + 1.627)
        + 0.0975 * np.sin(4.0 * alpha + 0.3764)
        + 0.1496 * np.sin(5.0 * alpha - 1.227)
    )
    cd = (
        0.5732
        - 0.3223 * np.sin(alpha - 35.64)
        + 0.2519 * np.sin(2.0 * alpha - 7.843)
        + 0.0639 * np.sin(3.0 * alpha + 2.251)
        - 0.0386 * np.sin(4.0 * alpha + 1.191)
        - 0.0371 * np.sin(5.0 * alpha + 5.581)
    )
    cm = (
        -0.0209
        - 0.3978 * np.sin(alpha + 0.2454)
        + 0.1364 * np.sin(2.0 * alpha + 0.6146)
    )
    return cl, cd, cm


def unsteady_factor(v_ref_m_s: Any, v_x_m_s: Any) -> np.ndarray:
    """Published Eq. (S3): ``Cus = 1 + 3.2 atan(0.5 vref / vx)``."""

    v_ref = np.asarray(v_ref_m_s, dtype=float)
    v_x = np.asarray(v_x_m_s, dtype=float)
    if np.any(v_ref < 0.0) or np.any(v_x < 0.0):
        raise ValueError("v_ref and v_x must be non-negative speeds")
    ratio = np.divide(
        0.5 * v_ref,
        v_x,
        out=np.full(np.broadcast_shapes(v_ref.shape, v_x.shape), np.inf),
        where=np.broadcast_to(v_x, np.broadcast_shapes(v_ref.shape, v_x.shape)) > 0.0,
    )
    ratio = np.where((v_ref == 0.0) & (v_x == 0.0), 0.0, ratio)
    return 1.0 + 3.2 * np.arctan(ratio)


def _normalise(vector: np.ndarray, *, name: str) -> np.ndarray:
    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    if np.any(norm <= 1.0e-12):
        raise ValueError(f"degenerate {name} vector")
    return vector / norm


def _deformed_point(
    x_m: np.ndarray,
    y_m: np.ndarray,
    z_m: np.ndarray,
    time_s: np.ndarray,
    *,
    omega_rad_s: float,
    flap_amplitude_rad: float,
    twist_amplitude_rad: float,
    twist_phase_rad: float,
    half_span_m: float,
) -> np.ndarray:
    """Apply the frozen RoboEagle twist followed by root-axis flapping."""

    theta = flap_amplitude_rad * np.sin(omega_rad_s * time_s)
    psi = (
        twist_amplitude_rad
        * (y_m / half_span_m)
        * np.sin(omega_rad_s * time_s + twist_phase_rad)
    )
    axis_x = robogeom.axis_x(y_m, half_span_m)

    cos_psi = np.cos(psi)
    sin_psi = np.sin(psi)
    x_twisted = axis_x + (x_m - axis_x) * cos_psi - z_m * sin_psi
    z_twisted = (x_m - axis_x) * sin_psi + z_m * cos_psi

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    y_flapped = y_m * cos_theta - z_twisted * sin_theta
    z_flapped = y_m * sin_theta + z_twisted * cos_theta
    return np.stack((x_twisted, y_flapped, z_flapped), axis=-1)


def simulate_case(case: RoboFalcon2Case) -> dict[str, Any]:
    """Cycle-average the published strip forces for one 184-grid condition."""

    case.validate()
    ns = int(case.ns)
    total_steps = int(case.steps_per_cycle) * int(case.n_cycle)
    period_s = 1.0 / float(case.frequency_hz)
    omega = 2.0 * np.pi * float(case.frequency_hz)

    # Paper Eq. (3): dy = y_ss / ns.  RoboEagle has no fold coordinate in the
    # 184 contract, so y_ss is the fixed half-span.
    dy = float(case.half_span_m) / ns
    y_strip = (np.arange(ns, dtype=float) + 0.5) * dy
    chord = np.asarray(robogeom.chord_at(y_strip), dtype=float)

    time = (
        np.arange(total_steps, dtype=float)
        * period_s
        / float(case.steps_per_cycle)
    )[:, None]
    y = y_strip[None, :]
    zeros = np.zeros_like(y)
    flap_amplitude = np.radians(case.flap_amplitude_deg)
    # Benchmark contract: nominal twist is peak-to-peak, hence solver half
    # amplitude is nominal/2 and remains spanwise-linear.
    twist_amplitude = np.radians(case.nominal_twist_deg / 2.0)
    twist_phase = np.radians(case.twist_phase_deg)

    kwargs = dict(
        omega_rad_s=omega,
        flap_amplitude_rad=flap_amplitude,
        twist_amplitude_rad=twist_amplitude,
        twist_phase_rad=twist_phase,
        half_span_m=float(case.half_span_m),
    )
    leading_edge = _deformed_point(zeros, y, zeros, time, **kwargs)
    trailing_edge = _deformed_point(chord[None, :], y, zeros, time, **kwargs)

    # The reference point in the paper is normally the leading edge.  A
    # centered geometric derivative supplies its body-relative flapping speed.
    derivative_dt = period_s * 1.0e-6
    leading_plus = _deformed_point(
        zeros, y, zeros, time + derivative_dt, **kwargs
    )
    leading_minus = _deformed_point(
        zeros, y, zeros, time - derivative_dt, **kwargs
    )
    reference_velocity = (leading_plus - leading_minus) / (2.0 * derivative_dt)

    chord_vector = trailing_edge - leading_edge

    # The straight leading edge defines the local span direction.  A centered
    # derivative is used so twist-gradient and flap geometry are retained.
    span_eps = min(0.25 * dy, 1.0e-4)
    y_plus = np.minimum(y + span_eps, float(case.half_span_m))
    y_minus = np.maximum(y - span_eps, 0.0)
    span_plus = _deformed_point(zeros, y_plus, zeros, time, **kwargs)
    span_minus = _deformed_point(zeros, y_minus, zeros, time, **kwargs)
    span_hat = _normalise(span_plus - span_minus, name="span")

    chord_orthogonal = chord_vector - (
        np.sum(chord_vector * span_hat, axis=-1, keepdims=True) * span_hat
    )
    chord_hat = _normalise(chord_orthogonal, name="chord")
    normal_hat = _normalise(np.cross(chord_hat, span_hat), name="normal")

    aoa = np.radians(case.aoa_deg)
    air_velocity = np.array(
        [case.airspeed_m_s, 0.0, case.airspeed_m_s * np.tan(aoa)],
        dtype=float,
    )
    relative_velocity = air_velocity[None, None, :] - reference_velocity
    section_velocity = relative_velocity - (
        np.sum(relative_velocity * span_hat, axis=-1, keepdims=True) * span_hat
    )
    v_loc = np.linalg.norm(section_velocity, axis=-1)
    if np.any(v_loc <= 1.0e-10):
        raise ValueError("local section speed reached zero")
    flow_hat = section_velocity / v_loc[..., None]

    alpha_loc = np.arctan2(
        np.sum(section_velocity * normal_hat, axis=-1),
        np.sum(section_velocity * chord_hat, axis=-1),
    )
    v_ref = np.linalg.norm(reference_velocity, axis=-1)
    v_x = np.abs(relative_velocity[..., 0])
    cus = unsteady_factor(v_ref, v_x)
    cl, cd, cm = force_coefficients(alpha_loc)

    # Paper Eq. (S2), evaluated independently on every strip and time sample.
    dynamic_factor = 0.5 * float(case.rho_kg_m3) * v_loc**2 * cus
    strip_area = chord[None, :] * dy
    d_lift = dynamic_factor * cl * strip_area
    d_drag = dynamic_factor * cd * strip_area
    d_moment = dynamic_factor * cm * chord[None, :] ** 2 * dy

    lift_hat = _normalise(np.cross(flow_hat, span_hat), name="lift")
    strip_force = d_lift[..., None] * lift_hat + d_drag[..., None] * flow_hat
    # The computed geometry is one half-wing.  Symmetric x/z forces and pitch
    # moments add; lateral forces cancel.
    body_force = 2.0 * np.sum(strip_force, axis=1)
    pitch_moment = 2.0 * np.sum(d_moment, axis=1)

    flow_axis = np.array([np.cos(aoa), 0.0, np.sin(aoa)], dtype=float)
    wind_lift_axis = np.array([-np.sin(aoa), 0.0, np.cos(aoa)], dtype=float)
    upwind_thrust_axis = -flow_axis
    lift_history = body_force @ wind_lift_axis
    thrust_history = body_force @ upwind_thrust_axis

    result = {
        "L": float(np.mean(lift_history)),
        "T": float(np.mean(thrust_history)),
        "M": float(np.mean(pitch_moment)),
        "diagnostics": {
            "alpha_loc_deg_min": float(np.degrees(np.min(alpha_loc))),
            "alpha_loc_deg_max": float(np.degrees(np.max(alpha_loc))),
            "alpha_loc_deg_mean": float(np.degrees(np.mean(alpha_loc))),
            "cus_min": float(np.min(cus)),
            "cus_max": float(np.max(cus)),
            "cus_mean": float(np.mean(cus)),
            "v_loc_m_s_mean": float(np.mean(v_loc)),
            "v_ref_m_s_mean": float(np.mean(v_ref)),
            "single_wing_area_m2": float(np.sum(strip_area)),
            "samples": total_steps,
        },
        "case": asdict(case),
    }
    scalar_values = [result["L"], result["T"], result["M"]]
    scalar_values.extend(result["diagnostics"].values())
    if not all(np.isfinite(value) for value in scalar_values):
        raise FloatingPointError("RoboFalcon 2.0 result contains non-finite values")
    return result


__all__ = [
    "DEFAULT_FLAP_AMPLITUDE_DEG",
    "DEFAULT_HALF_SPAN_M",
    "DEFAULT_RHO_KG_M3",
    "DEFAULT_TWIST_PHASE_DEG",
    "PAPER_DOI",
    "RoboFalcon2Case",
    "force_coefficients",
    "simulate_case",
    "unsteady_factor",
]
