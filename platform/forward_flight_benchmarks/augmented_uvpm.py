"""Exploratory UVLM-preserving load augmentation shared by two benchmarks.

The solver keeps the complete prescribed-wake UVLM solution and records an
explicit force ledger for its Kutta--Joukowski/circulatory and unsteady
Bernoulli contributions. For a periodic result, only the alternating part of
those two UVLM-owned contributions is transformed::

    F_aug = mean(F_uvlm) + K_AM (F_AM - mean(F_AM))
                           + G_H (F_C - mean(F_C)) + F_polar

``K_AM`` is the finite-aspect-ratio added-mass factor quoted by Izraelevitz et
al. (2017), and ``G_H`` is their Eq. (42) lifting-surface correction. The
cycle mean remains owned by UVLM. ``F_polar`` is the geometry-only nonlinear
polar residual in :mod:`uvlm_polar_correction`; it activates only beyond the
shared attached-incidence gate.

This module has no paper/case identifier and uses no measured-force residual.
It is deliberately a periodic two-pass load model, not yet a causal transient
production solver and not a reconstruction of Yang's PLEV/AWS implementation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pterasoftware as ps
from pterasoftware import _transformations

from fluxvortex.solver import UVPMHybridSolver

from .uvlm_polar_correction import _periodic_resample


@dataclass(frozen=True)
class PeriodicUVLMAugmentationParameters:
    """Constants copied from the source model, with no observation fit."""

    hoerner_surface_constant: float = 13.5
    added_mass_factor_at_ar3: float = 0.85
    added_mass_factor_at_ar6: float = 0.95

    def manifest(self) -> dict[str, float | str]:
        out: dict[str, float | str] = asdict(self)
        out.update(
            surface_gain_source="Izraelevitz et al. 2017 Eq. (42)",
            added_mass_source=(
                "Izraelevitz et al. 2017 Eqs. (35)-(36): K_AM=0.85 at AR=3 "
                "and 0.95 at AR=6; linear interpolation with endpoint clamp"
            ),
            observation_fit="none",
            scope="periodic two-pass alternating-load augmentation",
        )
        return out


DEFAULT_AUGMENTATION_PARAMETERS = PeriodicUVLMAugmentationParameters()


def hoerner_lifting_surface_gain(
    aspect_ratio: float,
    parameters: PeriodicUVLMAugmentationParameters = DEFAULT_AUGMENTATION_PARAMETERS,
) -> float:
    """Return the lifting-surface/lifting-line gain from paper Eq. (42)."""

    if aspect_ratio <= 0.0:
        raise ValueError("aspect ratio must be positive")
    numerator = 1.0 / (2.0 * np.pi) + 1.0 / (np.pi * aspect_ratio)
    denominator = numerator + (
        parameters.hoerner_surface_constant
        * np.pi
        / (180.0 * aspect_ratio**2)
    )
    return float(numerator / denominator)


def added_mass_aspect_ratio_factor(
    aspect_ratio: float,
    parameters: PeriodicUVLMAugmentationParameters = DEFAULT_AUGMENTATION_PARAMETERS,
) -> float:
    """Interpolate the paper's two finite-AR added-mass landmarks."""

    if aspect_ratio <= 0.0:
        raise ValueError("aspect ratio must be positive")
    fraction = np.clip((aspect_ratio - 3.0) / 3.0, 0.0, 1.0)
    return float(
        parameters.added_mass_factor_at_ar3
        + fraction
        * (
            parameters.added_mass_factor_at_ar6
            - parameters.added_mass_factor_at_ar3
        )
    )


class AugmentedUVPMHybridSolver(UVPMHybridSolver):
    """UVPM hybrid solver that records an auditable per-step UVLM load ledger."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._record_vpm_particles = bool(kwargs.pop("record_vpm_particles", True))
        super().__init__(*args, **kwargs)
        self.uvlm_load_ledger: list[dict[str, Any]] = []

    def _populate_next_airplanes_wake(self) -> None:
        if self._record_vpm_particles:
            super()._populate_next_airplanes_wake()
            return
        # Exact load-equivalent fast path.  The current FluxV particles are
        # one-way diagnostics and never enter the UVLM AIC or load calculation.
        self._prescribed_wake = True
        self._populate_next_airplanes_wake_vortex_points()
        self._populate_next_airplanes_wake_vortices()

    def _calculate_loads(self) -> None:
        current_strengths = np.asarray(
            self._current_bound_vortex_strengths, dtype=float
        ).copy()
        previous_strengths = np.asarray(
            self._last_bound_vortex_strengths, dtype=float
        ).copy()
        panel_areas = np.asarray(self.panel_areas, dtype=float).copy()
        panel_normals = np.asarray(self.stackUnitNormals_GP1, dtype=float).copy()

        super()._calculate_loads()

        unsteady_panel_forces_gp1 = -(
            self.current_operating_point.rho
            * (current_strengths - previous_strengths)[:, None]
            * panel_areas[:, None]
            * panel_normals
            / self.delta_time
        )
        transform = self.current_operating_point.T_pas_GP1_CgP1_to_W_CgP1
        airplane_rows: list[dict[str, np.ndarray]] = []
        global_position = 0
        for airplane in self.current_airplanes:
            total_gp1 = np.zeros(3, dtype=float)
            unsteady_gp1 = np.zeros(3, dtype=float)
            for wing in airplane.wings:
                panels = list(np.ravel(wing.panels))
                count = len(panels)
                total_gp1 += np.sum(
                    np.asarray([panel.forces_GP1 for panel in panels], dtype=float),
                    axis=0,
                )
                unsteady_gp1 += np.sum(
                    unsteady_panel_forces_gp1[
                        global_position : global_position + count
                    ],
                    axis=0,
                )
                global_position += count
            total_w = _transformations.apply_T_to_vectors(
                transform, total_gp1, has_point=False
            )
            unsteady_w = _transformations.apply_T_to_vectors(
                transform, unsteady_gp1, has_point=False
            )
            airplane_rows.append(
                {
                    "total_force_w_n": np.asarray(total_w, dtype=float),
                    "unsteady_force_w_n": np.asarray(unsteady_w, dtype=float),
                    "circulatory_force_w_n": np.asarray(
                        total_w - unsteady_w, dtype=float
                    ),
                }
            )
        self.uvlm_load_ledger.append(
            {"step": int(self._current_step), "airplanes": airplane_rows}
        )


def _coherent_cycle_rows(
    solver: AugmentedUVPMHybridSolver,
    *,
    movement: Any,
    period_s: float,
    airplane_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]]:
    rows = solver.uvlm_load_ledger
    if not rows:
        raise FloatingPointError("solver produced no UVLM load ledger")
    steps_per_cycle = int(round(period_s / movement.delta_time))
    last_step = int(rows[-1]["step"])
    cycle_at_last = last_step * movement.delta_time / period_s
    cycle_end = (
        last_step
        if np.isclose(cycle_at_last, round(cycle_at_last), atol=1.0e-10)
        else last_step + 1
    )
    cycle_start = cycle_end - steps_per_cycle
    selected = [row for row in rows if cycle_start <= row["step"] < cycle_end]
    if len(selected) != steps_per_cycle:
        raise FloatingPointError("load ledger does not contain one coherent cycle")
    steps = np.asarray([row["step"] for row in selected], dtype=int)
    phase = np.mod(steps * movement.delta_time / period_s, 1.0)
    total = np.asarray(
        [row["airplanes"][airplane_index]["total_force_w_n"] for row in selected]
    )
    unsteady = np.asarray(
        [row["airplanes"][airplane_index]["unsteady_force_w_n"] for row in selected]
    )
    circulatory = np.asarray(
        [row["airplanes"][airplane_index]["circulatory_force_w_n"] for row in selected]
    )
    return phase, total, unsteady, circulatory, [cycle_start, cycle_end - 1]


def extract_periodic_augmented_history(
    solver: AugmentedUVPMHybridSolver,
    movement: Any,
    *,
    period_s: float,
    rho_kg_m3: float,
    freestream_m_s: float,
    area_m2: float,
    aspect_ratio: float,
    polar_residual: dict[str, Any],
    output_samples: int = 128,
    airplane_index: int = 0,
    parameters: PeriodicUVLMAugmentationParameters = DEFAULT_AUGMENTATION_PARAMETERS,
) -> dict[str, Any]:
    """Transform one coherent cycle and add an aligned nonlinear polar residual."""

    phase, total, unsteady, circulatory, step_range = _coherent_cycle_rows(
        solver,
        movement=movement,
        period_s=period_s,
        airplane_index=airplane_index,
    )
    component_identity = np.max(np.abs(total - unsteady - circulatory))
    gain_unsteady = added_mass_aspect_ratio_factor(aspect_ratio, parameters)
    gain_circulatory = hoerner_lifting_surface_gain(aspect_ratio, parameters)
    transformed = (
        np.mean(total, axis=0)
        + gain_unsteady * (unsteady - np.mean(unsteady, axis=0))
        + gain_circulatory * (circulatory - np.mean(circulatory, axis=0))
    )
    target_phase, force_w = _periodic_resample(phase, transformed, output_samples)
    _, baseline_force_w = _periodic_resample(phase, total, output_samples)
    residual_phase = np.asarray(polar_residual["phase"], dtype=float)
    if target_phase.shape != residual_phase.shape or not np.allclose(
        target_phase, residual_phase, atol=1.0e-12, rtol=0.0
    ):
        raise ValueError("UVLM ledger and polar residual phases are not aligned")

    base_lift = -force_w[:, 2]
    base_drag = -force_w[:, 0]
    lift = base_lift + np.asarray(polar_residual["delta_lift_n"], dtype=float)
    drag = base_drag + np.asarray(polar_residual["delta_drag_n"], dtype=float)
    q_area = 0.5 * rho_kg_m3 * freestream_m_s**2 * area_m2
    mean_total = np.mean(total, axis=0)
    mean_lift = -mean_total[2] + float(polar_residual["mean_delta_lift_n"])
    mean_drag = -mean_total[0] + float(polar_residual["mean_delta_drag_n"])
    return {
        "phase": target_phase,
        "lift_n": lift,
        "drag_n": drag,
        "thrust_n": -drag,
        "CL": lift / q_area,
        "CD": drag / q_area,
        "CT": -drag / q_area,
        "mean_lift_n": float(mean_lift),
        "mean_drag_n": float(mean_drag),
        "mean_thrust_n": float(-mean_drag),
        "mean_CL": float(mean_lift / q_area),
        "mean_CD": float(mean_drag / q_area),
        "mean_CT": float(-mean_drag / q_area),
        "baseline_uvlm_lift_n": -baseline_force_w[:, 2],
        "baseline_uvlm_drag_n": -baseline_force_w[:, 0],
        "baseline_uvlm_thrust_n": baseline_force_w[:, 0],
        "baseline_uvlm_CL": -baseline_force_w[:, 2] / q_area,
        "baseline_uvlm_CD": -baseline_force_w[:, 0] / q_area,
        "baseline_uvlm_CT": baseline_force_w[:, 0] / q_area,
        "baseline_uvlm_mean_lift_n": float(-mean_total[2]),
        "baseline_uvlm_mean_drag_n": float(-mean_total[0]),
        "baseline_uvlm_mean_thrust_n": float(mean_total[0]),
        "component_identity_max_abs_n": float(component_identity),
        "source_cycle_step_range": step_range,
        "unsteady_ac_gain": gain_unsteady,
        "circulatory_ac_gain": gain_circulatory,
        "parameters": parameters.manifest(),
        "polar_parameters": polar_residual["parameters"],
        "model_semantics": (
            "exploratory periodic FluxV augmentation: UVLM cycle mean retained; "
            "source-derived finite-AR gains applied to UVLM load components; "
            "shared nonlinear polar residual added at separated incidence"
        ),
    }


def run_augmented_fluxv(
    movement: Any,
    *,
    period_s: float,
    rho_kg_m3: float,
    freestream_m_s: float,
    area_m2: float,
    aspect_ratio: float,
    polar_residual: dict[str, Any],
    output_samples: int = 128,
    record_vpm_particles: bool = False,
) -> dict[str, Any]:
    """Run the load-ledger solver and return the augmented periodic history."""

    problem = ps.problems.UnsteadyProblem(movement=movement, only_final_results=False)
    solver = AugmentedUVPMHybridSolver(
        problem,
        max_particles=20000,
        stretch=False,
        free_wake=False,
        record_vpm_particles=record_vpm_particles,
    )
    solver.run(
        prescribed_wake=True,
        calculate_streamlines=False,
        show_progress=False,
    )
    out = extract_periodic_augmented_history(
        solver,
        movement,
        period_s=period_s,
        rho_kg_m3=rho_kg_m3,
        freestream_m_s=freestream_m_s,
        area_m2=area_m2,
        aspect_ratio=aspect_ratio,
        polar_residual=polar_residual,
        output_samples=output_samples,
    )
    out["particle_count"] = int(solver._vpm_field.np)
    out["vpm_particle_fast_path"] = not record_vpm_particles
    return out


def blend_periodic_ullt_state_shape(
    ullt_history: dict[str, Any],
    separated_uvlm_history: dict[str, Any],
    separation_fraction: np.ndarray,
    *,
    rho_kg_m3: float,
    freestream_m_s: float,
    area_m2: float,
) -> dict[str, Any]:
    """Blend attached ULLT and separated UVLM *alternating* load shapes.

    The separated UVLM history owns the cycle mean.  The ULLT state response
    owns the attached alternating load, while the UVLM/polar response owns the
    separated alternating load.  Recentring the blended alternating component
    prevents a phase-dependent gate from silently moving the mean-force owner.
    This is a periodic reduced-order construction; it is not causal online
    switching and is labelled accordingly in the returned metadata.
    """

    phase = np.asarray(ullt_history["phase"], dtype=float)
    other_phase = np.asarray(separated_uvlm_history["phase"], dtype=float)
    if phase.shape != other_phase.shape or not np.allclose(
        phase, other_phase, atol=1.0e-12, rtol=0.0
    ):
        raise ValueError("ULLT and separated UVLM phases are not aligned")
    separation = np.asarray(separation_fraction, dtype=float)
    if separation.shape != phase.shape or np.any(~np.isfinite(separation)):
        raise ValueError("separation fraction must be a finite phase vector")
    if np.any((separation < 0.0) | (separation > 1.0)):
        raise ValueError("separation fraction must lie in [0, 1]")

    attached = np.column_stack(
        (
            np.asarray(ullt_history["lift_n"], dtype=float),
            np.asarray(ullt_history["drag_n"], dtype=float),
        )
    )
    separated = np.column_stack(
        (
            np.asarray(separated_uvlm_history["lift_n"], dtype=float),
            np.asarray(separated_uvlm_history["drag_n"], dtype=float),
        )
    )
    attached_ac = attached - np.mean(attached, axis=0)
    separated_ac = separated - np.mean(separated, axis=0)
    mixed_ac = (
        (1.0 - separation[:, None]) * attached_ac
        + separation[:, None] * separated_ac
    )
    mixed_ac -= np.mean(mixed_ac, axis=0)
    target_mean = np.array(
        [
            float(separated_uvlm_history["mean_lift_n"]),
            float(separated_uvlm_history["mean_drag_n"]),
        ]
    )
    loads = target_mean + mixed_ac
    q_area = 0.5 * rho_kg_m3 * freestream_m_s**2 * area_m2
    return {
        "phase": phase.copy(),
        "lift_n": loads[:, 0],
        "drag_n": loads[:, 1],
        "thrust_n": -loads[:, 1],
        "CL": loads[:, 0] / q_area,
        "CD": loads[:, 1] / q_area,
        "CT": -loads[:, 1] / q_area,
        "mean_lift_n": float(target_mean[0]),
        "mean_drag_n": float(target_mean[1]),
        "mean_thrust_n": float(-target_mean[1]),
        "mean_CL": float(target_mean[0] / q_area),
        "mean_CD": float(target_mean[1] / q_area),
        "mean_CT": float(-target_mean[1] / q_area),
        "separation_fraction": separation.copy(),
        "model_semantics": (
            "exploratory periodic FluxV v2: UVLM/polar cycle mean retained; "
            "one-state ULLT owns attached AC shape and UVLM/polar owns separated "
            "AC shape; observation-free shared gate; not an online transient solver"
        ),
    }
