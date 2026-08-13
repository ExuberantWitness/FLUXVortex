"""Razak--Dimitriadis rigid-wing experiment and frozen FluxV-v4b transfer.

The experiment was published by Razak and Dimitriadis (2014); the open-access
Lambert et al. (2017) paper supplies the geometry, test matrix, load-processing
contract, and vector phase-load plots used here.  Lambert's numerical runs used
measured 64-point motion histories which are not public.  This module therefore
uses an extrema-matched nominal sinusoid and never phase-fits it to the loads.

The two physical wings are kinematically symmetric.  One aerodynamic wing is
modelled and normalized by its own ``b*c`` area, matching Lambert's sectional
coefficient equations; the same coefficient would result from doubling both
load and reference area if opposite-wing aerodynamic interaction is neglected.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pterasoftware as ps

from .causal_incidence_owner import causal_incidence_persistence
from .ldvm_uvlm_correction import (
    LDVMSectionSettings,
    LESPThreshold,
    project_ldvm_delta_to_finite_wing,
    run_ldvm_separation_pair,
)
from .ptera_adapter import run_model
from .uvlm_polar_correction import (
    DEFAULT_POLAR_PARAMETERS,
    augment_uvlm_history,
    movement_polar_residual,
)


@dataclass(frozen=True)
class Razak2014Case:
    figure: int
    motion_family: str
    freestream_m_s: float
    frequency_hz: float
    pitch_min_deg: float
    pitch_max_deg: float
    chord_m: float = 0.160
    span_m: float = 0.400
    root_offset_m: float = 0.150
    flap_amplitude_deg: float = 30.0
    pitch_axis_fraction_chord: float = 0.25
    rho_kg_m3: float = 1.225
    nu_m2_s: float = 1.506e-5

    def __post_init__(self) -> None:
        if self.motion_family not in {"pitch_leading", "pitch_lagging"}:
            raise ValueError("motion family must be pitch_leading or pitch_lagging")
        if self.pitch_max_deg < self.pitch_min_deg:
            raise ValueError("pitch extrema are reversed")
        if min(self.freestream_m_s, self.frequency_hz, self.chord_m, self.span_m) <= 0:
            raise ValueError("flow, frequency, chord and span must be positive")

    @property
    def case_id(self) -> str:
        return f"razak2014_fig{self.figure}"

    @property
    def period_s(self) -> float:
        return 1.0 / self.frequency_hz

    @property
    def pitch_center_deg(self) -> float:
        return 0.5 * (self.pitch_min_deg + self.pitch_max_deg)

    @property
    def pitch_amplitude_deg(self) -> float:
        return 0.5 * (self.pitch_max_deg - self.pitch_min_deg)

    @property
    def pitch_phase_deg(self) -> float:
        # gamma=gamma_a sin(wt).  Leading pitch is +cos(wt); lagging is -cos(wt).
        return 90.0 if self.motion_family == "pitch_leading" else -90.0

    @property
    def area_m2(self) -> float:
        return self.chord_m * self.span_m

    @property
    def aspect_ratio(self) -> float:
        return self.span_m**2 / self.area_m2

    @property
    def reynolds(self) -> float:
        return self.freestream_m_s * self.chord_m / self.nu_m2_s

    @property
    def reduced_frequency(self) -> float:
        return np.pi * self.frequency_hz * self.chord_m / self.freestream_m_s

    def manifest(self) -> dict[str, Any]:
        out = asdict(self)
        out.update(
            case_id=self.case_id,
            period_s=self.period_s,
            pitch_center_deg=self.pitch_center_deg,
            pitch_amplitude_deg=self.pitch_amplitude_deg,
            pitch_phase_deg=self.pitch_phase_deg,
            area_m2=self.area_m2,
            aspect_ratio=self.aspect_ratio,
            reynolds=self.reynolds,
            reduced_frequency=self.reduced_frequency,
        )
        return out


RAZAK_2014_CASES = {
    case.figure: case
    for case in (
        Razak2014Case(9, "pitch_leading", 6.0, 0.79, -4.0, 8.0),
        Razak2014Case(10, "pitch_leading", 6.0, 1.23, -10.0, 2.0),
        Razak2014Case(11, "pitch_leading", 14.8, 1.23, -8.0, 4.0),
        Razak2014Case(13, "pitch_lagging", 9.4, 1.23, -5.0, 7.0),
        Razak2014Case(14, "pitch_lagging", 6.0, 1.50, -12.0, 0.0),
        Razak2014Case(15, "pitch_lagging", 6.0, 1.50, 4.0, 16.0),
    )
}


def _quality_settings(quality: str) -> tuple[int, int, int, int, int]:
    if quality == "smoke":
        return 3, 5, 32, 2, 1
    if quality == "full":
        # Four chordwise panels resolve the UVLM pressure load without the
        # accidental 14-chordwise-panel cubic solve used by the first aborted
        # pilot.  Twelve cosine-spaced spanwise panels retain the published
        # finite-wing geometry at 128 steps/cycle.
        return 4, 12, 128, 3, 2
    raise ValueError("quality must be smoke or full")


def build_razak_movement(
    case: Razak2014Case,
    quality: str = "full",
    *,
    settings: tuple[int, int, int, int, int] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Build the rectangular NACA6409 wing with nominal harmonic motion."""

    nc, ns, steps_per_cycle, cycles, wake_cycles = (
        _quality_settings(quality) if settings is None else settings
    )
    if min(nc, ns, steps_per_cycle, cycles, wake_cycles) <= 0:
        raise ValueError("all discretization settings must be positive")
    if wake_cycles > cycles:
        raise ValueError("wake cycles cannot exceed simulated cycles")

    airfoil = ps.geometry.airfoil.Airfoil(name="naca6409")
    root = ps.geometry.wing_cross_section.WingCrossSection(
        airfoil=airfoil,
        chord=case.chord_m,
        num_spanwise_panels=ns,
        spanwise_spacing="cosine",
    )
    tip = ps.geometry.wing_cross_section.WingCrossSection(
        airfoil=airfoil,
        chord=case.chord_m,
        Lp_Wcsp_Lpp=(0.0, case.span_m, 0.0),
        num_spanwise_panels=None,
        spanwise_spacing=None,
    )
    wing = ps.geometry.wing.Wing(
        name=f"Razak 2014 Figure {case.figure} right wing",
        wing_cross_sections=[root, tip],
        Ler_Gs_Cgs=(0.0, case.root_offset_m, 0.0),
        angles_Gs_to_Wn_ixyz=(0.0, case.pitch_center_deg, 0.0),
        symmetric=False,
        num_chordwise_panels=nc,
        chordwise_spacing="uniform",
    )
    airplane = ps.geometry.airplane.Airplane(
        wings=[wing],
        name=case.case_id,
        s_ref=case.area_m2,
        c_ref=case.chord_m,
        b_ref=case.span_m,
    )
    section_movements = [
        ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
            base_wing_cross_section=section
        )
        for section in (root, tip)
    ]
    pivot_m = case.pitch_axis_fraction_chord * case.chord_m
    wing_movement = ps.movements.wing_movement.WingMovement(
        base_wing=wing,
        wing_cross_section_movements=section_movements,
        ampAngles_Gs_to_Wn_ixyz=(
            case.flap_amplitude_deg,
            case.pitch_amplitude_deg,
            0.0,
        ),
        periodAngles_Gs_to_Wn_ixyz=(case.period_s, case.period_s, 0.0),
        phaseAngles_Gs_to_Wn_ixyz=(0.0, case.pitch_phase_deg, 0.0),
        rotationPointOffset_Gs_Ler=(pivot_m, -case.root_offset_m, 0.0),
    )
    airplane_movement = ps.movements.airplane_movement.AirplaneMovement(
        base_airplane=airplane,
        wing_movements=[wing_movement],
    )
    operating_point = ps.operating_point.OperatingPoint(
        rho=case.rho_kg_m3,
        vCg__E=case.freestream_m_s,
        alpha=0.0,
        beta=0.0,
        nu=case.nu_m2_s,
    )
    movement = ps.movements.movement.Movement(
        airplane_movements=[airplane_movement],
        operating_point_movement=ps.movements.operating_point_movement.OperatingPointMovement(
            base_operating_point=operating_point
        ),
        delta_time=case.period_s / steps_per_cycle,
        num_cycles=cycles,
        max_wake_cycles=wake_cycles,
    )
    metadata = {
        **case.manifest(),
        "grid_chord_span": [nc, ns],
        "steps_per_cycle": steps_per_cycle,
        "cycles": cycles,
        "max_wake_cycles": wake_cycles,
        "airfoil_geometry": "Ptera NACA6409 mean-camber surface; 9% thickness not resolved",
        "wing_count_adapter": (
            "one of two symmetric experimental wings; coefficients use one-wing bc area "
            "and neglect opposite-wing aerodynamic interaction"
        ),
        "pitch_axis_adapter": (
            "quarter chord; the open Lambert summary does not publish the rig pitch-axis "
            "coordinate, so this is a declared reconstruction assumption"
        ),
        "motion_adapter": (
            "extrema-matched harmonic flap/pitch; not the unpublished measured 64-point "
            "cycle-average history used by Lambert's VLM"
        ),
    }
    return movement, metadata


def run_razak_old_fluxv(
    case: Razak2014Case,
    quality: str = "full",
    *,
    output_samples: int = 128,
    settings: tuple[int, int, int, int, int] | None = None,
) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    movement, metadata = build_razak_movement(case, quality, settings=settings)
    result = run_model(
        movement,
        "fluxv_uvpm",
        period_s=case.period_s,
        rho=case.rho_kg_m3,
        speed=case.freestream_m_s,
        area=case.area_m2,
        output_samples=output_samples,
    )
    result["drag_n"] = -np.asarray(result["thrust_n"])
    result["CD"] = -np.asarray(result["CT"])
    result["mean_drag_n"] = -float(result["mean_thrust_n"])
    result["mean_CD"] = -float(result["mean_CT"])
    return result, movement, metadata


def _periodic_derivative(values: np.ndarray, step: float) -> np.ndarray:
    return (np.roll(values, -1) - np.roll(values, 1)) / (2.0 * step)


def apply_frozen_v4b_transfer(
    case: Razak2014Case,
    baseline: dict[str, Any],
    movement: Any,
    *,
    output_samples: int = 128,
    ldvm_steps_per_cycle: int = 512,
    ldvm_max_wake_steps: int = 256,
) -> dict[str, Any]:
    """Apply the committed v4b mechanism without force-observation fitting."""

    source_range = baseline["source_cycle_step_range"]
    polar = movement_polar_residual(
        movement,
        source_cycle_step_range=source_range,
        period_s=case.period_s,
        freestream_m_s=case.freestream_m_s,
        rho_kg_m3=case.rho_kg_m3,
        aspect_ratio=case.aspect_ratio,
        output_samples=output_samples,
        parameters=DEFAULT_POLAR_PARAMETERS,
    )
    persistent = augment_uvlm_history(
        baseline,
        polar,
        rho_kg_m3=case.rho_kg_m3,
        freestream_m_s=case.freestream_m_s,
        area_m2=case.area_m2,
    )
    if ldvm_steps_per_cycle < output_samples:
        raise ValueError("LDVM integration cannot be coarser than the output history")
    if ldvm_max_wake_steps < 8:
        raise ValueError("LDVM wake retention must be at least eight steps")
    alpha_eff = np.asarray(polar["alpha_rad"], dtype=float)
    phase = np.asarray(polar["phase"], dtype=float)
    delta_time_convective = (
        case.freestream_m_s * case.period_s / case.chord_m / output_samples
    )
    warmup_cycles = 12
    owner = causal_incidence_persistence(
        np.tile(alpha_eff, (warmup_cycles, 1)),
        delta_time_convective=delta_time_convective,
        strip_weights=np.asarray(polar["mean_strip_area_m2"], dtype=float),
    )
    persistence = np.asarray(owner["global_persistence"])[-output_samples:]

    omega = 2.0 * np.pi
    ldvm_phase = np.arange(ldvm_steps_per_cycle, dtype=float) / ldvm_steps_per_cycle
    pitch_internal = np.deg2rad(case.pitch_center_deg) + np.deg2rad(
        case.pitch_amplitude_deg
    ) * np.sin(omega * ldvm_phase + np.deg2rad(case.pitch_phase_deg))
    ldvm_delta_time_convective = (
        case.freestream_m_s * case.period_s / case.chord_m / ldvm_steps_per_cycle
    )
    pitch_rate_internal = _periodic_derivative(
        pitch_internal, ldvm_delta_time_convective
    )
    # Use the same quarter-chord incidence as the frozen polar ledger.  The
    # inferred plunge ratio is the unique value satisfying
    # alpha_eff=theta+atan2(-h_dot/U,1) for the clean-room LDVM convention.
    alpha_internal = np.empty((ldvm_steps_per_cycle, alpha_eff.shape[1]))
    for strip_index in range(alpha_eff.shape[1]):
        alpha_internal[:, strip_index] = np.interp(
            ldvm_phase,
            phase,
            alpha_eff[:, strip_index],
            period=1.0,
        )
    heave_rate_internal = -np.tan(alpha_internal - pitch_internal[:, None])
    threshold = LESPThreshold(
        value=float(np.sin(np.deg2rad(10.31))),
        section_family="NACA 6409",
        reynolds=case.reynolds,
        source=(
            "Lambert et al. 2017 Figure 7b Kirchhoff alpha1=10.31 deg; "
            "mapped before scoring as Lcrit=sin(alpha1)"
        ),
        source_role="published static-separation parameter mapping hypothesis",
    )
    strip_area = np.asarray(polar["mean_strip_area_m2"], dtype=float)
    weights = strip_area / np.sum(strip_area)
    delta_cl = np.zeros(output_samples)
    delta_cd = np.zeros(output_samples)
    shedding = np.zeros(output_samples)
    for strip_index, weight in enumerate(weights):
        alpha_two = np.tile(pitch_internal, 2)
        rate_two = np.tile(pitch_rate_internal, 2)
        heave_two = np.tile(heave_rate_internal[:, strip_index], 2)
        pair = run_ldvm_separation_pair(
            alpha_rad=alpha_two,
            alpha_rate_per_convective_time=rate_two,
            heave_rate_over_u=heave_two,
            delta_time_convective=ldvm_delta_time_convective,
            pivot_fraction_chord=case.pitch_axis_fraction_chord,
            threshold=threshold,
            settings=LDVMSectionSettings(
                ndiv=32,
                naterm=14,
                max_wake_steps=ldvm_max_wake_steps,
            ),
        )
        selected = slice(ldvm_steps_per_cycle, 2 * ldvm_steps_per_cycle)
        projection = project_ldvm_delta_to_finite_wing(
            np.asarray(pair["delta"]["CNc"])[selected],
            np.asarray(pair["delta"]["CNnc"])[selected],
            np.asarray(pair["delta"]["CNnonl"])[selected],
            np.asarray(pair["delta"]["CSf"])[selected],
            pitch_internal,
            aspect_ratio=case.aspect_ratio,
        )
        internal_delta_cl = np.asarray(projection["delta_CL"]) * np.cos(
            np.deg2rad(case.flap_amplitude_deg) * np.sin(omega * ldvm_phase)
        )
        internal_delta_cd = np.asarray(projection["delta_CD"])
        internal_shedding = np.asarray(pair["shed_lev"], dtype=float)[selected]
        delta_cl += weight * np.interp(phase, ldvm_phase, internal_delta_cl, period=1.0)
        delta_cd += weight * np.interp(phase, ldvm_phase, internal_delta_cd, period=1.0)
        shedding += weight * np.interp(phase, ldvm_phase, internal_shedding, period=1.0)

    old_cl = np.asarray(baseline["CL"], dtype=float)
    old_cd = np.asarray(baseline["CD"], dtype=float)
    polar_cl = np.asarray(persistent["CL"], dtype=float)
    polar_cd = np.asarray(persistent["CD"], dtype=float)
    transient_cl = old_cl + delta_cl
    transient_cd = old_cd + delta_cd
    v4_cl = (1.0 - persistence) * transient_cl + persistence * polar_cl
    v4_cd = (1.0 - persistence) * transient_cd + persistence * polar_cd
    q_area = 0.5 * case.rho_kg_m3 * case.freestream_m_s**2 * case.area_m2
    return {
        "phase": phase,
        "CL": v4_cl,
        "CD": v4_cd,
        "lift_n": q_area * v4_cl,
        "drag_n": q_area * v4_cd,
        "thrust_n": -q_area * v4_cd,
        "mean_CL": float(np.mean(v4_cl)),
        "mean_CD": float(np.mean(v4_cd)),
        "mean_lift_n": float(q_area * np.mean(v4_cl)),
        "mean_drag_n": float(q_area * np.mean(v4_cd)),
        "mean_thrust_n": -float(q_area * np.mean(v4_cd)),
        "persistence": persistence,
        "ldvm_delta_CL": delta_cl,
        "ldvm_delta_CD": delta_cd,
        "ldvm_shedding": shedding,
        "lesp_critical": threshold.value,
        "ldvm_steps_per_cycle": int(ldvm_steps_per_cycle),
        "ldvm_max_wake_steps": int(ldvm_max_wake_steps),
        "ldvm_delta_time_convective": float(ldvm_delta_time_convective),
        "lesp_provenance": threshold.manifest(),
        "polar_parameters": polar["parameters"],
        "model_semantics": (
            "frozen v4b transfer: retained old FluxV UVLM, paired-LDVM "
            "discrepancy, and causal persistent full-angle-polar ownership"
        ),
        "limitations": [
            "nominal harmonic rather than unpublished measured 64-point motion",
            "independent 2-D LDVM strips, not a conservative coupled 3-D LEV row",
            "quarter-chord pitch axis is a declared reconstruction assumption",
            "opposite-wing aerodynamic interaction is omitted",
            "LDVM integration uses a separate 512-step default and a 256-step "
            "wake-retention cap after the direct 128-step transfer exposed a "
            "singular newborn-vortex constraint; both require sensitivity checks",
        ],
    }


__all__ = [
    "RAZAK_2014_CASES",
    "Razak2014Case",
    "apply_frozen_v4b_transfer",
    "build_razak_movement",
    "run_razak_old_fluxv",
]
