"""Baik 2012 W1--W4 pitching--plunging flat-plate benchmark.

The primary force source is Baik's dissertation, Figures 5.24--5.27.  Those
figures restore the steady hydrodynamic force that the earlier AIAA precursor
removed with its pre-trigger reference; they are therefore the corrected-total
load histories used for scoring.

The wide-Strouhal motions do *not* use harmonic plunge displacement.  Baik
enforces a sinusoidal plunge-induced incidence,

``atan(-h_dot/U) = alpha_pl,max sin(2 pi t/T)``,

and obtains ``h`` by periodic integration.  The helpers below preserve this
kinematic contract without fitting the published force histories.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
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
class Baik2012Case:
    """One wide-Strouhal case from dissertation Table 1.4."""

    case_id: str
    strouhal: float
    reduced_frequency: float
    heave_to_chord: float
    pitch_amplitude_deg: float
    period_s: float
    chord_m: float = 0.076
    span_m: float = 0.600
    thickness_to_chord: float = 0.0625
    mean_alpha_deg: float = 8.0
    effective_alpha_amplitude_deg: float = 14.0
    pivot_fraction_chord: float = 0.25
    reynolds: float = 5_000.0
    rho_kg_m3: float = 998.2

    def __post_init__(self) -> None:
        if not self.case_id.startswith("W"):
            raise ValueError("Baik case id must start with W")
        positive = (
            self.strouhal,
            self.reduced_frequency,
            self.heave_to_chord,
            self.pitch_amplitude_deg,
            self.period_s,
            self.chord_m,
            self.span_m,
            self.thickness_to_chord,
            self.reynolds,
            self.rho_kg_m3,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("Baik case parameters must be finite and positive")

    @property
    def frequency_hz(self) -> float:
        return 1.0 / self.period_s

    @property
    def freestream_m_s(self) -> float:
        # Use the source's measured period rather than an assumed water
        # viscosity.  k=pi f c/U defines the velocity actually used by the
        # kinematic experiment; Re then determines the implied nu below.
        return np.pi * self.frequency_hz * self.chord_m / self.reduced_frequency

    @property
    def nu_m2_s(self) -> float:
        return self.freestream_m_s * self.chord_m / self.reynolds

    @property
    def area_m2(self) -> float:
        return self.chord_m * self.span_m

    @property
    def kinematic_strouhal(self) -> float:
        """Return the unrounded Strouhal number implied by ``k`` and ``h0``."""

        return 2.0 * self.reduced_frequency * self.heave_to_chord / np.pi

    @property
    def experimental_filter_harmonic(self) -> int:
        """Highest cycle harmonic retained by the source's 1 Hz low-pass."""

        return int(np.floor(1.0 / self.frequency_hz + 1.0e-12))

    @property
    def geometric_aspect_ratio(self) -> float:
        return self.span_m / self.chord_m

    @property
    def peak_plunge_induced_alpha_deg(self) -> float:
        return float(
            np.rad2deg(
                _solve_peak_plunge_alpha_rad(
                    self.reduced_frequency * self.heave_to_chord
                )
            )
        )

    @property
    def implemented_pitch_amplitude_deg(self) -> float:
        # Table 1.4 rounds theta0 to two decimals.  The implemented value is
        # derived from the exact displacement constraint and the prescribed
        # 14-degree effective-incidence amplitude, leaving both source laws
        # internally consistent.  The printed value remains in the manifest.
        return self.peak_plunge_induced_alpha_deg - self.effective_alpha_amplitude_deg

    @property
    def thickness_m(self) -> float:
        return self.thickness_to_chord * self.chord_m

    @property
    def rounded_edge_radius_m(self) -> float:
        return 0.5 * self.thickness_m

    def manifest(self) -> dict[str, Any]:
        out = asdict(self)
        out.update(
            frequency_hz=self.frequency_hz,
            freestream_m_s=self.freestream_m_s,
            nu_m2_s=self.nu_m2_s,
            area_m2=self.area_m2,
            kinematic_strouhal=self.kinematic_strouhal,
            experimental_filter_harmonic=self.experimental_filter_harmonic,
            geometric_aspect_ratio=self.geometric_aspect_ratio,
            peak_plunge_induced_alpha_deg=self.peak_plunge_induced_alpha_deg,
            implemented_pitch_amplitude_deg=self.implemented_pitch_amplitude_deg,
            published_pitch_amplitude_rounding_delta_deg=(
                self.implemented_pitch_amplitude_deg - self.pitch_amplitude_deg
            ),
            thickness_m=self.thickness_m,
            rounded_leading_edge_radius_m=self.rounded_edge_radius_m,
            rounded_trailing_edge_radius_m=self.rounded_edge_radius_m,
        )
        return out


BAIK_2012_CASES = {
    case.case_id: case
    for case in (
        Baik2012Case("W1", 0.16, 0.5, 0.50, 13.16, 7.13),
        Baik2012Case("W2", 0.32, 1.0, 0.50, 33.73, 3.56),
        Baik2012Case("W3", 0.16, 1.0, 0.25, 13.16, 3.56),
        Baik2012Case("W4", 0.32, 0.5, 1.00, 33.73, 7.13),
    )
}


@lru_cache(maxsize=None)
def _solve_peak_plunge_alpha_rad(k_times_h0: float) -> float:
    """Solve Baik Eq. (1.12) without introducing a SciPy dependency.

    With ``xi=2*pi*t/T``, the quarter-stroke displacement constraint is
    ``integral_0^(pi/2) tan(A*sin(xi)) dxi = 2*k*h0``.
    """

    if not np.isfinite(k_times_h0) or k_times_h0 <= 0.0:
        raise ValueError("k*h0 must be finite and positive")
    nodes, weights = np.polynomial.legendre.leggauss(128)
    xi = 0.25 * np.pi * (nodes + 1.0)
    quadrature_weights = 0.25 * np.pi * weights
    target = 2.0 * k_times_h0

    def residual(angle: float) -> float:
        return float(np.sum(quadrature_weights * np.tan(angle * np.sin(xi))) - target)

    lower = 0.0
    upper = np.deg2rad(89.0)
    for _ in range(80):
        midpoint = 0.5 * (lower + upper)
        if residual(midpoint) < 0.0:
            lower = midpoint
        else:
            upper = midpoint
    return 0.5 * (lower + upper)


def _periodic_cumulative_integral(values: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """Return a periodic primitive with its value at phase zero set to zero."""

    value = np.asarray(values, dtype=float)
    tau = np.asarray(phase, dtype=float)
    if value.shape != tau.shape or value.ndim != 1 or value.size < 8:
        raise ValueError("periodic integral inputs must be aligned 1-D histories")
    if np.any(np.diff(tau) <= 0.0) or tau[0] != 0.0 or tau[-1] >= 1.0:
        raise ValueError("phase must increase from zero without duplicating one")
    extended_tau = np.concatenate((tau, [1.0]))
    extended_value = np.concatenate((value, value[:1]))
    integral = np.zeros(extended_tau.size)
    integral[1:] = np.cumsum(
        0.5 * (extended_value[1:] + extended_value[:-1]) * np.diff(extended_tau)
    )
    # A tiny discrete drift can otherwise break the custom Ptera spacing's
    # exact return-to-zero contract. Remove it linearly over the period.
    integral -= extended_tau * integral[-1]
    return integral[:-1]


def baik_kinematics(
    phase: np.ndarray | float,
    case: Baik2012Case,
) -> dict[str, np.ndarray]:
    """Return source-faithful pitch, plunge, and incidence histories.

    Phase zero is the upper plunge limit and ``0 < phase < 0.5`` is downstroke.
    Positive ``h`` is upward.  Positive aerodynamic pitch is nose-up.
    """

    tau = np.mod(np.asarray(phase, dtype=float), 1.0)
    omega_tau = 2.0 * np.pi * tau
    pitch_deg = -case.implemented_pitch_amplitude_deg * np.sin(omega_tau)
    alpha_plunge_rad = np.deg2rad(case.peak_plunge_induced_alpha_deg) * np.sin(
        omega_tau
    )
    heave_rate_over_u = -np.tan(alpha_plunge_rad)
    effective_alpha_deg = (
        case.mean_alpha_deg
        + pitch_deg
        + np.rad2deg(np.arctan2(-heave_rate_over_u, 1.0))
    )
    return {
        "pitch_deg": pitch_deg,
        "geometric_alpha_deg": case.mean_alpha_deg + pitch_deg,
        "alpha_plunge_deg": np.rad2deg(alpha_plunge_rad),
        "effective_alpha_deg": effective_alpha_deg,
        "heave_rate_over_u": heave_rate_over_u,
    }


@lru_cache(maxsize=None)
def _heave_spacing_samples(
    case_id: str, samples: int = 16_384
) -> tuple[np.ndarray, ...]:
    case = BAIK_2012_CASES[case_id]
    phase = np.arange(samples, dtype=float) / samples
    velocity = baik_kinematics(phase, case)["heave_rate_over_u"]
    # dh/dtau = T*U*(h_dot/U). Start at the upper limit h=+h0*c.
    displacement = _periodic_cumulative_integral(
        case.period_s * case.freestream_m_s * velocity,
        phase,
    )
    displacement -= 0.5 * (np.max(displacement) + np.min(displacement))
    amplitude = 0.5 * np.ptp(displacement)
    if amplitude <= 0.0:
        raise FloatingPointError("Baik heave integration produced zero amplitude")
    normalized = displacement / amplitude
    # The source rounds theta0 and T, so amplitude agreement is close but not
    # algebraically exact. Ptera scales this shape by the printed h0*c.
    return phase, normalized, displacement


def baik_heave_spacing(case_id: str):
    """Return a Ptera custom spacing for the source's non-harmonic plunge."""

    phase_grid, normalized, _ = _heave_spacing_samples(case_id)

    def spacing(phase_rad: np.ndarray) -> np.ndarray:
        # Ptera validates that a custom spacing starts at zero.  Shift the
        # physical displacement by -pi/2 inside the callable, then supply a
        # +90-degree phase lead in WingMovement so the generated motion starts
        # at Baik's upper plunge limit.
        phase = (
            np.mod(np.asarray(phase_rad, dtype=float), 2.0 * np.pi) / (2.0 * np.pi)
            - 0.25
        ) % 1.0
        return np.interp(phase, phase_grid, normalized, period=1.0)

    spacing.__name__ = f"baik_{case_id.lower()}_sinusoidal_incidence_heave"
    return spacing


def _thin_symmetric_airfoil() -> Any:
    # UVLM consumes only the mean camber line. This tiny symmetric outline is
    # explicitly not a model of the physical 6.25%-thick rounded plate.
    eps = 1.0e-4
    outline = np.asarray([[1.0, eps], [0.5, eps], [0.0, 0.0], [0.5, -eps], [1.0, -eps]])
    return ps.geometry.airfoil.Airfoil(
        name="Baik-2012-zero-camber-mean-surface-adapter",
        outline_A_lp=outline,
        resample=False,
    )


def _quality_settings(quality: str) -> tuple[int, int, int, int]:
    if quality == "smoke":
        return 2, 8, 32, 2
    if quality == "full":
        # The experiment is wall-to-wall/end-plated, whereas Ptera has no wall
        # boundary.  Keep the physical 600 mm span as a declared free-tip
        # adapter and spend resolution on the non-harmonic time history.  A
        # separate span/time sensitivity run is required before interpreting
        # small old-to-v4b differences.
        return 4, 8, 128, 3
    raise ValueError("quality must be smoke or full")


def build_baik_movement(
    case: Baik2012Case,
    quality: str = "full",
    *,
    settings: tuple[int, int, int, int] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Build the declared free-tip surrogate for the quasi-2D experiment."""

    nc, ns, steps_per_cycle, cycles = (
        _quality_settings(quality) if settings is None else settings
    )
    if min(nc, ns, steps_per_cycle, cycles) <= 0:
        raise ValueError("all discretization settings must be positive")
    airfoil = _thin_symmetric_airfoil()
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
    pivot = case.pivot_fraction_chord * case.chord_m
    wing = ps.geometry.wing.Wing(
        name=f"Baik 2012 {case.case_id} end-plated rectangular adapter",
        wing_cross_sections=[root, tip],
        Ler_Gs_Cgs=(-pivot, 0.0, 0.0),
        angles_Gs_to_Wn_ixyz=(0.0, case.mean_alpha_deg, 0.0),
        symmetric=False,
        num_chordwise_panels=nc,
        chordwise_spacing="uniform",
    )
    airplane = ps.geometry.airplane.Airplane(
        wings=[wing],
        name=f"Baik 2012 {case.case_id}",
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
    wing_movement = ps.movements.wing_movement.WingMovement(
        base_wing=wing,
        wing_cross_section_movements=section_movements,
        ampLer_Gs_Cgs=(0.0, 0.0, case.heave_to_chord * case.chord_m),
        periodLer_Gs_Cgs=(0.0, 0.0, case.period_s),
        spacingLer_Gs_Cgs=("sine", "sine", baik_heave_spacing(case.case_id)),
        # Custom spacing itself starts at the upper limit; no extra phase shift.
        phaseLer_Gs_Cgs=(0.0, 0.0, 90.0),
        ampAngles_Gs_to_Wn_ixyz=(0.0, case.implemented_pitch_amplitude_deg, 0.0),
        periodAngles_Gs_to_Wn_ixyz=(0.0, case.period_s, 0.0),
        # Ptera uses sine spacing; +180 degrees gives -sin.
        phaseAngles_Gs_to_Wn_ixyz=(0.0, 180.0, 0.0),
        rotationPointOffset_Gs_Ler=(pivot, 0.0, 0.0),
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
        max_wake_cycles=min(2, cycles),
    )
    phase_grid, normalized, raw_displacement = _heave_spacing_samples(case.case_id)
    raw_amplitude = 0.5 * np.ptp(raw_displacement)
    return movement, {
        "case": case.manifest(),
        "grid_chord_span": [nc, ns],
        "requested_steps_per_cycle": steps_per_cycle,
        "actual_steps_per_cycle": case.period_s / movement.delta_time,
        "cycles": cycles,
        "max_wake_cycles": min(2, cycles),
        "motion": (
            "theta=-theta0*sin(2*pi*t/T); h_dot/U=-tan(alpha_pl,max*"
            "sin(2*pi*t/T)); h(0)=+h0*c"
        ),
        "phase_origin": "upper plunge limit; 0<t/T<0.5 is downstroke",
        "integrated_raw_heave_amplitude_m": float(raw_amplitude),
        "printed_heave_amplitude_m": case.heave_to_chord * case.chord_m,
        "integrated_to_printed_amplitude_ratio": float(
            raw_amplitude / (case.heave_to_chord * case.chord_m)
        ),
        "spacing_extrema": [float(np.min(normalized)), float(np.max(normalized))],
        "airfoil_adapter": airfoil.name,
        "thickness_limitation": (
            "physical t/c=6.25% rounded plate; UVLM represents only its zero-camber "
            "mean surface and cannot resolve thickness/viscous edge effects"
        ),
        "span_adapter": (
            "one full 600 mm free-tip surrogate for the wall-to-wall/end-plated "
            "quasi-2D experiment; Ptera has no wall-image/endplate boundary"
        ),
    }


def run_baik_old_fluxv(
    case: Baik2012Case,
    quality: str = "full",
    *,
    output_samples: int = 128,
    settings: tuple[int, int, int, int] | None = None,
) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    movement, metadata = build_baik_movement(case, quality, settings=settings)
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


def sharp_fourier_lowpass(
    values: np.ndarray,
    *,
    maximum_harmonic: int,
) -> np.ndarray:
    """Apply the experiment's ideal one-cycle Fourier low-pass.

    ``values`` must be uniformly sampled over one cycle without a duplicate
    endpoint.  The DC term and harmonics through ``maximum_harmonic`` are
    preserved exactly; all higher resolved harmonics are set to zero.
    """

    history = np.asarray(values, dtype=float)
    if history.ndim != 1 or history.size < 8 or not np.isfinite(history).all():
        raise ValueError("values must be a finite one-dimensional cycle")
    if not isinstance(maximum_harmonic, int) or maximum_harmonic < 0:
        raise ValueError("maximum_harmonic must be a non-negative integer")
    spectrum = np.fft.rfft(history)
    spectrum[maximum_harmonic + 1 :] = 0.0
    return np.fft.irfft(spectrum, n=history.size)


def apply_declared_v4b_transfer(
    case: Baik2012Case,
    baseline: dict[str, Any],
    movement: Any,
    *,
    output_samples: int = 128,
    ldvm_steps_per_cycle: int = 512,
    ldvm_max_wake_steps: int = 256,
    lesp_critical: float = 0.11,
) -> dict[str, Any]:
    """Apply v4b using a provenance-labelled flat-plate LESP hypothesis.

    Ramesh's flat-plate case at Re=1000 uses ``Lcrit=0.11`` throughout the
    detailed body text and plots, while Table 4.1 prints 0.19.  The body-text
    value is the declared primary transfer; 0.19 is retained only as a
    source-conflict sensitivity.  Neither value is fitted to Baik's loads.
    """

    source_range = baseline["source_cycle_step_range"]
    polar = movement_polar_residual(
        movement,
        source_cycle_step_range=source_range,
        period_s=case.period_s,
        freestream_m_s=case.freestream_m_s,
        rho_kg_m3=case.rho_kg_m3,
        aspect_ratio=case.geometric_aspect_ratio,
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
        raise ValueError("LDVM integration cannot be coarser than output")
    if ldvm_max_wake_steps < 8:
        raise ValueError("LDVM wake retention must be at least eight steps")
    if not np.isfinite(lesp_critical) or lesp_critical <= 0.0:
        raise ValueError("LESP critical value must be finite and positive")

    phase = np.arange(output_samples, dtype=float) / output_samples
    alpha_effective = np.deg2rad(baik_kinematics(phase, case)["effective_alpha_deg"])
    delta_time_convective = (
        case.freestream_m_s * case.period_s / case.chord_m / output_samples
    )
    warmup_cycles = 12
    owner = causal_incidence_persistence(
        np.tile(alpha_effective, warmup_cycles),
        delta_time_convective=delta_time_convective,
    )
    persistence = np.asarray(owner["global_persistence"])[-output_samples:]

    internal_phase = np.arange(ldvm_steps_per_cycle, dtype=float) / ldvm_steps_per_cycle
    kinematics = baik_kinematics(internal_phase, case)
    alpha_internal = np.deg2rad(kinematics["geometric_alpha_deg"])
    dt_internal = (
        case.freestream_m_s * case.period_s / case.chord_m / ldvm_steps_per_cycle
    )
    alpha_rate_internal = _periodic_derivative(alpha_internal, dt_internal)
    heave_rate_internal = np.asarray(kinematics["heave_rate_over_u"], dtype=float)
    threshold = LESPThreshold(
        value=float(lesp_critical),
        section_family="rounded flat plate",
        reynolds=case.reynolds,
        source=(
            "Ramesh 2013 thesis flat-plate Re=1000: section 4.3.5 and "
            "Figures 4.19/4.21 use Lcrit=0.11 while Table 4.1 prints 0.19; "
            f"declared value={lesp_critical:g} is a frozen cross-Re/thickness "
            "transfer hypothesis, not a Baik force fit"
        ),
        source_role="published flat-plate transfer hypothesis; no Baik force fit",
    )
    cycles = 3
    ldvm_settings = LDVMSectionSettings(
        ndiv=32,
        naterm=14,
        max_wake_steps=ldvm_max_wake_steps,
        core_radius_chord=0.02,
    )
    pair = run_ldvm_separation_pair(
        alpha_rad=np.tile(alpha_internal, cycles),
        alpha_rate_per_convective_time=np.tile(alpha_rate_internal, cycles),
        heave_rate_over_u=np.tile(heave_rate_internal, cycles),
        delta_time_convective=dt_internal,
        pivot_fraction_chord=case.pivot_fraction_chord,
        threshold=threshold,
        settings=ldvm_settings,
    )
    selected = slice((cycles - 1) * ldvm_steps_per_cycle, cycles * ldvm_steps_per_cycle)
    projection = project_ldvm_delta_to_finite_wing(
        np.asarray(pair["delta"]["CNc"])[selected],
        np.asarray(pair["delta"]["CNnc"])[selected],
        np.asarray(pair["delta"]["CNnonl"])[selected],
        np.asarray(pair["delta"]["CSf"])[selected],
        alpha_internal,
        # Use the same physical-span/free-tip adapter as the retained UVLM and
        # polar ledgers.  Setting only this term to infinite AR would mix two
        # incompatible boundary conditions inside one force balance.
        aspect_ratio=case.geometric_aspect_ratio,
    )
    delta_cl = np.interp(
        phase, internal_phase, np.asarray(projection["delta_CL"]), period=1.0
    )
    delta_cd = np.interp(
        phase, internal_phase, np.asarray(projection["delta_CD"]), period=1.0
    )
    shedding = np.interp(
        phase,
        internal_phase,
        (
            np.abs(np.asarray(pair["separated"]["lesp"], dtype=float)[selected])
            > threshold.value
        ).astype(float),
        period=1.0,
    )
    old_cl = np.asarray(baseline["CL"], dtype=float)
    old_cd = np.asarray(baseline["CD"], dtype=float)
    polar_cl = np.asarray(persistent["CL"], dtype=float)
    polar_cd = np.asarray(persistent["CD"], dtype=float)
    transient_cl = old_cl + delta_cl
    transient_cd = old_cd + delta_cd
    corrected_cl = (1.0 - persistence) * transient_cl + persistence * polar_cl
    corrected_cd = (1.0 - persistence) * transient_cd + persistence * polar_cd
    q_area = 0.5 * case.rho_kg_m3 * case.freestream_m_s**2 * case.area_m2
    return {
        "phase": phase,
        "CL": corrected_cl,
        "CD": corrected_cd,
        "lift_n": q_area * corrected_cl,
        "drag_n": q_area * corrected_cd,
        "thrust_n": -q_area * corrected_cd,
        "mean_CL": float(np.mean(corrected_cl)),
        "mean_CD": float(np.mean(corrected_cd)),
        "mean_lift_n": float(q_area * np.mean(corrected_cl)),
        "mean_drag_n": float(q_area * np.mean(corrected_cd)),
        "mean_thrust_n": -float(q_area * np.mean(corrected_cd)),
        "persistence": persistence,
        "ldvm_delta_CL": delta_cl,
        "ldvm_delta_CD": delta_cd,
        "ldvm_shedding": shedding,
        "lesp_critical": threshold.value,
        "lesp_provenance": threshold.manifest(),
        "ldvm_steps_per_cycle": ldvm_steps_per_cycle,
        "ldvm_max_wake_steps": ldvm_max_wake_steps,
        "ldvm_delta_time_convective": dt_internal,
        "ldvm_settings": asdict(ldvm_settings),
        "polar_parameters": polar["parameters"],
        "model_semantics": (
            "declared v4b transfer: retained old FluxV UVLM, paired-LDVM "
            "discrepancy, and causal persistent full-angle-polar ownership"
        ),
        "limitations": [
            "Lcrit transfers a Re=1000, 2.3%-thick plate value to Baik Re=5000, 6.25%",
            "Ramesh Table 4.1/body-text Lcrit conflict is exposed as a sensitivity",
            "UVLM mean-surface adapter omits physical 6.25% thickness and viscosity",
            "wall/endplate boundary is approximated by a physical-span free-tip wing",
            "independent 2-D LDVM discrepancy is not a common-wake conservative coupling",
            "separate LDVM step and wake settings require one-factor sensitivity",
        ],
    }


__all__ = [
    "BAIK_2012_CASES",
    "Baik2012Case",
    "apply_declared_v4b_transfer",
    "baik_heave_spacing",
    "baik_kinematics",
    "build_baik_movement",
    "run_baik_old_fluxv",
    "sharp_fourier_lowpass",
]
