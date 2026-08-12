"""UVLM-preserving finite-wing full-angle polar residual.

The current :class:`fluxvortex.solver.UVPMHybridSolver` obtains its loads from
PteraSoftware's prescribed-wake UVLM.  This module leaves that unsteady UVLM
history intact and replaces only a geometry-derived linear, quasi-steady
finite-wing polar by a bounded full-angle polar::

    C_L,linear = a_3D * alpha
    C_L,full   = a_3D * sin(alpha) * cos(alpha)
    C_D,full   = C_D90 * sin(alpha)**2

    delta F = q dS [(C_L,full-C_L,linear) e_L + C_D,full e_D]

``a_3D`` is the Prandtl finite-wing slope obtained from ``a0=2*pi`` and the
actual aspect ratio.  ``C_D90=1.20`` is the finite-aspect-ratio flat-plate
constant already frozen in :mod:`platform.lb_static`; it is not fitted to
either benchmark.  A smooth 15--20 degree incidence gate leaves the attached
UVLM channel unchanged and introduces the residual only as the wing enters
the full-angle regime.  The correction accepts no paper or case identifier.

This is a load-ledger residual, not a reconstruction of Yang's PLEV/AWS
solver.  The UVLM continues to own circulation, wake memory, and non-circulatory
loads.  The residual is evaluated from the moving panel geometry at the UVLM
collocation points and therefore vanishes exactly at zero local incidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FullAnglePolarParameters:
    """Source-frozen constants for the shared nonlinear polar residual."""

    section_lift_slope_per_rad: float = 2.0 * np.pi
    span_efficiency: float = 1.0
    drag_coefficient_at_90_deg: float = 1.20
    section_velocity_reference_fraction_chord: float = 0.25
    attached_limit_deg: float = 15.0
    fully_separated_deg: float = 20.0

    def __post_init__(self) -> None:
        if self.section_lift_slope_per_rad <= 0.0:
            raise ValueError("section lift slope must be positive")
        if self.span_efficiency <= 0.0:
            raise ValueError("span efficiency must be positive")
        if self.drag_coefficient_at_90_deg < 0.0:
            raise ValueError("90-degree drag coefficient cannot be negative")
        if not 0.0 <= self.section_velocity_reference_fraction_chord <= 1.0:
            raise ValueError("section velocity reference must lie on the chord")
        if not 0.0 <= self.attached_limit_deg < self.fully_separated_deg <= 90.0:
            raise ValueError("invalid attached-to-separated angle interval")

    def manifest(self) -> dict[str, float | str]:
        out: dict[str, float | str] = asdict(self)
        out.update(
            lift_slope_source="thin-airfoil a0=2*pi plus Prandtl finite-wing correction",
            full_angle_lift_source="Nabawy-Crowther finite-wing normal-force form",
            drag_source=(
                "Hoerner finite-AR flat-plate CD90=1.20 frozen in lb_static.py; "
                "retained as a cross-AR approximation"
            ),
            velocity_policy_source=(
                "local incidence uses the moving strip point at the configured "
                "chord fraction; default is the quarter chord"
            ),
            transition_source=(
                "exploratory generic thin-wing 15-to-20 degree smooth stall interval; "
                "introduced after the v0 diagnostic, shared by both papers, and not "
                "fitted per case"
            ),
            observation_fit="none",
        )
        return out


DEFAULT_POLAR_PARAMETERS = FullAnglePolarParameters()


def finite_wing_lift_slope(
    aspect_ratio: float,
    parameters: FullAnglePolarParameters = DEFAULT_POLAR_PARAMETERS,
) -> float:
    """Return the Prandtl finite-wing lift slope in inverse radians."""

    if aspect_ratio <= 0.0:
        raise ValueError("aspect ratio must be positive")
    a0 = parameters.section_lift_slope_per_rad
    return float(a0 / (1.0 + a0 / (np.pi * parameters.span_efficiency * aspect_ratio)))


def full_angle_polar_residual_coefficients(
    alpha_rad: np.ndarray | float,
    aspect_ratio: float,
    parameters: FullAnglePolarParameters = DEFAULT_POLAR_PARAMETERS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(delta_CL, delta_CD)`` relative to the linear finite-wing polar."""

    alpha = np.asarray(alpha_rad, dtype=float)
    slope = finite_wing_lift_slope(aspect_ratio, parameters)
    alpha_abs_deg = np.abs(np.rad2deg(alpha))
    blend = np.clip(
        (alpha_abs_deg - parameters.attached_limit_deg)
        / (parameters.fully_separated_deg - parameters.attached_limit_deg),
        0.0,
        1.0,
    )
    blend = blend**2 * (3.0 - 2.0 * blend)
    delta_cl = blend * slope * (np.sin(alpha) * np.cos(alpha) - alpha)
    delta_cd = blend * parameters.drag_coefficient_at_90_deg * np.sin(alpha) ** 2
    return delta_cl, delta_cd


def _unit(vector: np.ndarray, *, floor: float = 1.0e-12) -> np.ndarray:
    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    return vector / np.maximum(norm, floor)


def _strip_geometry(
    airplane: Any,
    *,
    reference_fraction_chord: float,
) -> tuple[np.ndarray, ...]:
    centres: list[np.ndarray] = []
    chords: list[np.ndarray] = []
    spans: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    areas: list[float] = []
    for wing in airplane.wings:
        panels = wing.panels
        for span_index in range(panels.shape[1]):
            leading = panels[0, span_index]
            trailing = panels[-1, span_index]
            left_le = np.asarray(leading.Flpp_G_Cg, dtype=float)
            right_le = np.asarray(leading.Frpp_G_Cg, dtype=float)
            left_te = np.asarray(trailing.Blpp_G_Cg, dtype=float)
            right_te = np.asarray(trailing.Brpp_G_Cg, dtype=float)
            reference_fraction = reference_fraction_chord
            left_reference = left_le + reference_fraction * (left_te - left_le)
            right_reference = right_le + reference_fraction * (right_te - right_le)
            le_mid = 0.5 * (left_le + right_le)
            te_mid = 0.5 * (left_te + right_te)
            strip_panels = panels[:, span_index]
            strip_areas = np.asarray(
                [float(panel.area) for panel in strip_panels], dtype=float
            )
            strip_normals = np.asarray(
                [np.asarray(panel.unitNormal_G, dtype=float) for panel in strip_panels]
            )
            centres.append(0.5 * (left_reference + right_reference))
            chords.append(te_mid - le_mid)
            spans.append(right_reference - left_reference)
            normals.append(
                np.sum(strip_areas[:, None] * strip_normals, axis=0)
                / np.sum(strip_areas)
            )
            areas.append(float(np.sum(strip_areas)))
    if not centres:
        raise ValueError("movement airplane contains no aerodynamic panels")
    return (
        np.asarray(centres),
        _unit(np.asarray(chords)),
        _unit(np.asarray(spans)),
        _unit(np.asarray(normals)),
        np.asarray(areas),
    )


def _periodic_resample(
    phase: np.ndarray,
    values: np.ndarray,
    output_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    if output_samples < 2:
        raise ValueError("output_samples must be at least two")
    target = np.arange(output_samples, dtype=float) / output_samples
    order = np.argsort(phase)
    out = np.empty((output_samples,) + values.shape[1:], dtype=float)
    flat = values.reshape(values.shape[0], -1)
    flat_out = out.reshape(output_samples, -1)
    for column in range(flat.shape[1]):
        flat_out[:, column] = np.interp(
            target,
            phase[order],
            flat[order, column],
            period=1.0,
        )
    return target, out


def movement_polar_residual(
    movement: Any,
    *,
    source_cycle_step_range: tuple[int, int] | list[int],
    period_s: float,
    freestream_m_s: float,
    rho_kg_m3: float,
    aspect_ratio: float,
    output_samples: int = 128,
    parameters: FullAnglePolarParameters = DEFAULT_POLAR_PARAMETERS,
) -> dict[str, Any]:
    """Evaluate the shared polar residual on one coherent movement cycle.

    ``source_cycle_step_range`` is inclusive and must match the raw cycle used
    by the baseline UVLM history.  Panel velocities use a periodic centred
    difference on exactly that cycle, avoiding a duplicated endpoint.
    """

    if period_s <= 0.0 or freestream_m_s <= 0.0 or rho_kg_m3 <= 0.0:
        raise ValueError("period, freestream, and density must be positive")
    start, stop = (int(value) for value in source_cycle_step_range)
    if stop < start:
        raise ValueError("invalid source cycle step range")
    airplanes = movement.airplanes[0]
    if start < 0 or stop >= len(airplanes):
        raise ValueError("source cycle step range lies outside movement history")
    step_ids = np.arange(start, stop + 1, dtype=int)
    expected = int(round(period_s / movement.delta_time))
    if step_ids.size != expected:
        raise ValueError(
            f"cycle has {step_ids.size} steps but period/delta_time gives {expected}"
        )

    geometry = [
        _strip_geometry(
            airplanes[step],
            reference_fraction_chord=(
                parameters.section_velocity_reference_fraction_chord
            ),
        )
        for step in step_ids
    ]
    strip_counts = {item[0].shape[0] for item in geometry}
    if len(strip_counts) != 1:
        raise ValueError("strip topology changes within the selected cycle")
    centres = np.stack([item[0] for item in geometry])
    chord_hat = np.stack([item[1] for item in geometry])
    span_hat = np.stack([item[2] for item in geometry])
    normal_hat = np.stack([item[3] for item in geometry])
    area_m2 = np.stack([item[4] for item in geometry])

    panel_velocity = (np.roll(centres, -1, axis=0) - np.roll(centres, 1, axis=0)) / (
        2.0 * movement.delta_time
    )
    relative_velocity = (
        np.array([freestream_m_s, 0.0, 0.0])[None, None, :] - panel_velocity
    )
    relative_velocity -= (
        np.sum(relative_velocity * span_hat, axis=-1, keepdims=True) * span_hat
    )
    speed = np.linalg.norm(relative_velocity, axis=-1)
    drag_hat = _unit(relative_velocity)
    chordwise_velocity = np.sum(relative_velocity * chord_hat, axis=-1)
    normal_velocity = np.sum(relative_velocity * normal_hat, axis=-1)
    alpha_rad = np.arctan2(normal_velocity, chordwise_velocity)

    # Build a lift direction from the panel normal, made orthogonal to the
    # local relative-flow direction.  This remains consistently upward on
    # both halves of a symmetric Ptera wing, unlike a raw span cross product.
    lift_hat = normal_hat - (
        np.sum(normal_hat * drag_hat, axis=-1, keepdims=True) * drag_hat
    )
    lift_hat = _unit(lift_hat)
    delta_cl, delta_cd = full_angle_polar_residual_coefficients(
        alpha_rad,
        aspect_ratio,
        parameters,
    )
    dynamic_pressure = 0.5 * rho_kg_m3 * speed**2
    panel_force = (
        dynamic_pressure[..., None]
        * area_m2[..., None]
        * (delta_cl[..., None] * lift_hat + delta_cd[..., None] * drag_hat)
    )
    force_g = np.sum(panel_force, axis=1)
    # Unit constant-profile-drag load on the same moving strips.  This is kept
    # separate from the nonlinear polar residual so a source-published Cd0
    # (for example Scherer's Cd0=0.057 in Izraelevitz Figure 14) can be added
    # exactly once to any load history without changing the UVLM owner.
    unit_profile_force_g = np.sum(
        dynamic_pressure[..., None] * area_m2[..., None] * drag_hat,
        axis=1,
    )
    # ``force_g`` is an output-axis load vector: +z is positive lift and +x
    # is positive drag.  It is deliberately not Ptera's ``forces_W`` vector,
    # whose +z is down and +x is thrust in the adapters used here.

    raw_phase = np.mod(step_ids * movement.delta_time / period_s, 1.0)
    phase, force_resampled = _periodic_resample(raw_phase, force_g, output_samples)
    _, strip_force_resampled = _periodic_resample(
        raw_phase, panel_force, output_samples
    )
    force_resampled = np.sum(strip_force_resampled, axis=1)
    _, strip_area_resampled = _periodic_resample(raw_phase, area_m2, output_samples)
    _, unit_profile_resampled = _periodic_resample(
        raw_phase, unit_profile_force_g, output_samples
    )
    _, alpha_resampled = _periodic_resample(raw_phase, alpha_rad, output_samples)
    _, speed_resampled = _periodic_resample(raw_phase, speed, output_samples)
    # Resampling can move a discrete mean slightly when the source and target
    # grids have different sizes.  All downstream periodic ledgers therefore
    # use the returned phase-grid history as their single mean authority.
    mean_strip_force = np.mean(strip_force_resampled, axis=0)
    mean_force = np.sum(mean_strip_force, axis=0)
    mean_unit_profile_force = np.mean(unit_profile_resampled, axis=0)
    return {
        "phase": phase,
        "delta_force_g_n": force_resampled,
        "strip_delta_force_g_n": strip_force_resampled,
        "mean_strip_delta_force_g_n": mean_strip_force,
        "strip_area_m2": strip_area_resampled,
        "mean_strip_area_m2": np.mean(area_m2, axis=0),
        "delta_lift_n": force_resampled[:, 2],
        "delta_drag_n": force_resampled[:, 0],
        "mean_delta_force_g_n": mean_force,
        "mean_delta_lift_n": float(mean_force[2]),
        "mean_delta_drag_n": float(mean_force[0]),
        "unit_profile_drag_force_g_n": unit_profile_resampled,
        "unit_profile_drag_lift_n": unit_profile_resampled[:, 2],
        "unit_profile_drag_drag_n": unit_profile_resampled[:, 0],
        "mean_unit_profile_drag_force_g_n": mean_unit_profile_force,
        "mean_unit_profile_drag_lift_n": float(mean_unit_profile_force[2]),
        "mean_unit_profile_drag_drag_n": float(mean_unit_profile_force[0]),
        "alpha_rad": alpha_resampled,
        "relative_speed_m_s": speed_resampled,
        "max_abs_alpha_deg": float(np.max(np.abs(np.rad2deg(alpha_rad)))),
        "finite_wing_lift_slope_per_rad": finite_wing_lift_slope(
            aspect_ratio, parameters
        ),
        "strip_count": int(next(iter(strip_counts))),
        "source_cycle_step_range": [start, stop],
        "parameters": parameters.manifest(),
        "model_semantics": (
            "UVLM load history plus a geometry-only finite-wing full-angle "
            "polar residual; no observation-derived fit and not Yang PLEV/AWS"
        ),
    }


def add_constant_profile_drag(
    history: dict[str, Any],
    kinematic_residual: dict[str, Any],
    *,
    coefficient: float,
    rho_kg_m3: float,
    freestream_m_s: float,
    area_m2: float,
) -> dict[str, Any]:
    """Add a source-specified constant profile drag to a load history.

    The unit profile force is evaluated along the local two-dimensional
    relative-flow direction by :func:`movement_polar_residual`.  The helper
    neither re-runs nor replaces UVLM/ULLT; it only adds ``coefficient`` times
    that frozen kinematic load.  ``coefficient=0`` is an exact reduction.
    """

    if coefficient < 0.0 or not np.isfinite(coefficient):
        raise ValueError("profile drag coefficient must be finite and nonnegative")
    phase = np.asarray(history["phase"], dtype=float)
    residual_phase = np.asarray(kinematic_residual["phase"], dtype=float)
    if phase.shape != residual_phase.shape or not np.allclose(
        phase, residual_phase, atol=1.0e-12, rtol=0.0
    ):
        raise ValueError("history and profile-drag phases are not aligned")
    q_area = 0.5 * rho_kg_m3 * freestream_m_s**2 * area_m2
    if q_area <= 0.0:
        raise ValueError("reference dynamic pressure times area must be positive")

    profile_lift = coefficient * np.asarray(
        kinematic_residual["unit_profile_drag_lift_n"], dtype=float
    )
    profile_drag = coefficient * np.asarray(
        kinematic_residual["unit_profile_drag_drag_n"], dtype=float
    )
    lift = np.asarray(history["lift_n"], dtype=float) + profile_lift
    if "drag_n" in history:
        base_drag = np.asarray(history["drag_n"], dtype=float)
    else:
        base_drag = -np.asarray(history["thrust_n"], dtype=float)
    drag = base_drag + profile_drag

    mean_lift = float(
        history.get("mean_lift_n", np.mean(history["lift_n"]))
        + coefficient * kinematic_residual["mean_unit_profile_drag_lift_n"]
    )
    mean_drag = float(
        history.get("mean_drag_n", np.mean(base_drag))
        + coefficient * kinematic_residual["mean_unit_profile_drag_drag_n"]
    )
    out = dict(history)
    out.update(
        phase=phase.copy(),
        lift_n=lift,
        drag_n=drag,
        thrust_n=-drag,
        CL=lift / q_area,
        CD=drag / q_area,
        CT=-drag / q_area,
        mean_lift_n=mean_lift,
        mean_drag_n=mean_drag,
        mean_thrust_n=-mean_drag,
        mean_CL=mean_lift / q_area,
        mean_CD=mean_drag / q_area,
        mean_CT=-mean_drag / q_area,
        profile_drag_coefficient=float(coefficient),
        profile_drag_lift_n=profile_lift,
        profile_drag_drag_n=profile_drag,
        profile_drag_semantics=(
            "constant Cd0 times the moving-strip local dynamic-pressure load; "
            "added exactly once outside the UVLM/ULLT load owner"
        ),
    )
    return out


def augment_uvlm_history(
    baseline: dict[str, Any],
    residual: dict[str, Any],
    *,
    rho_kg_m3: float,
    freestream_m_s: float,
    area_m2: float,
) -> dict[str, Any]:
    """Add a polar residual to a resampled UVLM load history."""

    phase = np.asarray(baseline["phase"], dtype=float)
    residual_phase = np.asarray(residual["phase"], dtype=float)
    if phase.shape != residual_phase.shape or not np.allclose(
        phase, residual_phase, atol=1.0e-12, rtol=0.0
    ):
        raise ValueError("baseline and residual phases are not aligned")
    lift = np.asarray(baseline["lift_n"], dtype=float) + np.asarray(
        residual["delta_lift_n"], dtype=float
    )
    baseline_drag = -np.asarray(baseline["thrust_n"], dtype=float)
    drag = baseline_drag + np.asarray(residual["delta_drag_n"], dtype=float)
    q_area = 0.5 * rho_kg_m3 * freestream_m_s**2 * area_m2
    if q_area <= 0.0:
        raise ValueError("reference dynamic pressure times area must be positive")
    out = {
        "phase": phase.copy(),
        "lift_n": lift,
        "drag_n": drag,
        "thrust_n": -drag,
        "CL": lift / q_area,
        "CD": drag / q_area,
        "CT": -drag / q_area,
        "mean_lift_n": float(baseline["mean_lift_n"] + residual["mean_delta_lift_n"]),
        "mean_drag_n": float(
            -baseline["mean_thrust_n"] + residual["mean_delta_drag_n"]
        ),
        "mean_thrust_n": float(
            baseline["mean_thrust_n"] - residual["mean_delta_drag_n"]
        ),
        "mean_CL": float(
            (baseline["mean_lift_n"] + residual["mean_delta_lift_n"]) / q_area
        ),
        "mean_CD": float(
            (-baseline["mean_thrust_n"] + residual["mean_delta_drag_n"]) / q_area
        ),
        "mean_CT": float(
            (baseline["mean_thrust_n"] - residual["mean_delta_drag_n"]) / q_area
        ),
        "baseline_model": "fluxv_uvpm",
        "model_semantics": residual["model_semantics"],
        "polar_residual": residual,
    }
    return out
