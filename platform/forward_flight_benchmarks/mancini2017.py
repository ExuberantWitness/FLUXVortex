"""Mancini (2017) finite-wing pitch-and-hold experiment and frozen v4b bridge.

The experiment uses an aspect-ratio-four rectangular flat plate in water and
reports whole-wing lift during leading-edge pitch-up from 0 to 45 degrees.
This module reconstructs the published fast and slow motions without reading
the force traces.  The v4b bridge is the already-frozen Stevens transient
mechanism: retained finite-wing UVLM plus a separated-minus-attached LDVM
increment with the pre-existing thin-plate ``Lcrit=0.11`` transfer.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pterasoftware as ps

from fluxvortex.solver import UVPMHybridSolver

from .ldvm_uvlm_correction import (
    LDVMSectionSettings,
    LESPThreshold,
    project_ldvm_delta_to_finite_wing,
    run_ldvm_separation_pair,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
MANCINI_FIG4_13B_CSV = (
    REPO_ROOT / "docs/forward_flight_large_pitch/reproductions/"
    "mancini2017_v4b_20260820/source_data/"
    "mancini2017_fig4_13b_pitch_lift_digitized.csv"
)
MANCINI_FIG4_13B_CSV_SHA256 = (
    "8feecd8469bc5b7dff00c761d773d91464143e394c244cbfe0502a473df4db7c"
)
MANCINI_EXPERIMENTAL_OBSERVABLES = ("CL",)
MANCINI_EXPERIMENT_HAS_SPANWISE_LOADS = False
FROZEN_V4B_LESP_CRITICAL = 0.11


@dataclass(frozen=True)
class Mancini2017Case:
    case_id: str
    acceleration_distance_chords: float
    reduced_pitch_rate: float
    eldredge_smoothing: float
    chord_m: float = 0.0762
    aspect_ratio: float = 4.0
    thickness_fraction: float = 0.05
    reynolds: float = 20_000.0
    freestream_m_s: float = 0.26
    rho_kg_m3: float = 998.0
    initial_pitch_deg: float = 0.0
    maximum_pitch_deg: float = 45.0
    pivot_fraction_chord: float = 0.0
    warmup_chords: float = 2.0
    observation_chords: float = 5.0

    def __post_init__(self) -> None:
        if self.case_id not in {"fast_pitch", "slow_pitch"}:
            raise ValueError("Mancini case must be fast_pitch or slow_pitch")
        if (
            min(
                self.acceleration_distance_chords,
                self.eldredge_smoothing,
                self.chord_m,
                self.aspect_ratio,
                self.reynolds,
                self.freestream_m_s,
            )
            <= 0.0
        ):
            raise ValueError("Mancini geometry, flow and motion must be positive")
        if self.pivot_fraction_chord != 0.0:
            raise ValueError("the frozen Mancini cases pitch about the leading edge")

    @property
    def span_m(self) -> float:
        return self.aspect_ratio * self.chord_m

    @property
    def semispan_m(self) -> float:
        return 0.5 * self.span_m

    @property
    def area_m2(self) -> float:
        return self.span_m * self.chord_m

    @property
    def nu_m2_s(self) -> float:
        return self.freestream_m_s * self.chord_m / self.reynolds

    @property
    def convective_time_s(self) -> float:
        return self.chord_m / self.freestream_m_s

    @property
    def ideal_reduced_pitch_rate(self) -> float:
        return np.deg2rad(self.maximum_pitch_deg - self.initial_pitch_deg) / (
            2.0 * self.acceleration_distance_chords
        )

    @property
    def closure_start_chords(self) -> float:
        # Keep even the smooth Eldredge tail of the artificial closure outside
        # the physical observation window.  Four convective times are enough
        # for the slowest frozen smoothing value a=4 to decay below roundoff.
        return self.warmup_chords + self.observation_chords + 4.0

    @property
    def waveform_period_chords(self) -> float:
        return self.closure_start_chords + 2.0 * self.acceleration_distance_chords + 2.0

    @property
    def waveform_period_s(self) -> float:
        return self.waveform_period_chords * self.convective_time_s

    def manifest(self) -> dict[str, Any]:
        output = asdict(self)
        output.update(
            span_m=self.span_m,
            semispan_m=self.semispan_m,
            area_m2=self.area_m2,
            nu_m2_s=self.nu_m2_s,
            convective_time_s=self.convective_time_s,
            ideal_reduced_pitch_rate=self.ideal_reduced_pitch_rate,
            waveform_period_chords=self.waveform_period_chords,
            waveform_period_s=self.waveform_period_s,
        )
        return output


MANCINI_2017_CASES = {
    case.case_id: case
    for case in (
        Mancini2017Case("fast_pitch", 1.0, 0.39, 15.0),
        Mancini2017Case("slow_pitch", 6.0, 0.065, 4.0),
    )
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_mancini_fig4_13b_experiment(
    path: Path = MANCINI_FIG4_13B_CSV,
    *,
    verify_hash: bool = True,
) -> dict[str, Any]:
    digest = _sha256(path)
    if verify_hash and digest != MANCINI_FIG4_13B_CSV_SHA256:
        raise ValueError(
            "Mancini Figure 4.13b CSV hash mismatch: "
            f"expected {MANCINI_FIG4_13B_CSV_SHA256}, got {digest}"
        )
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        expected = (
            "t_star",
            "experiment_CL_fast_pitch",
            "experiment_CL_slow_pitch",
            "source_figure",
            "data_role",
        )
        if tuple(reader.fieldnames or ()) != expected:
            raise ValueError(
                f"unexpected Mancini digitization columns: {reader.fieldnames}"
            )
        rows = list(reader)
    if len(rows) != 1301:
        raise ValueError(f"expected 1301 Mancini samples, got {len(rows)}")
    t_star = np.asarray([float(row["t_star"]) for row in rows])
    fast = np.asarray([float(row["experiment_CL_fast_pitch"]) for row in rows])
    slow = np.asarray([float(row["experiment_CL_slow_pitch"]) for row in rows])
    if not np.allclose(t_star, np.linspace(0.0, 13.0, 1301), atol=1.0e-12):
        raise ValueError("Mancini digitization is not on the frozen 0.01 grid")
    if not np.all(np.isfinite(fast + slow)):
        raise ValueError("Mancini digitization contains non-finite values")
    if {row["source_figure"] for row in rows} != {"4.13b"}:
        raise ValueError("Mancini rows must all cite Figure 4.13b")
    if {row["data_role"] for row in rows} != {"digitized_experimental_curve"}:
        raise ValueError("Mancini data role drifted")
    return {
        "t_star": t_star,
        "CL_fast_pitch": fast,
        "CL_slow_pitch": slow,
        "observed_channels": MANCINI_EXPERIMENTAL_OBSERVABLES,
        "spanwise_loads_available": MANCINI_EXPERIMENT_HAS_SPANWISE_LOADS,
        "digitization_sha256": digest,
        "data_role": "digitized_experimental_whole_wing_lift_curve",
    }


def _stable_log_cosh(value: np.ndarray) -> np.ndarray:
    return np.logaddexp(value, -value) - np.log(2.0)


def _eldredge_step(
    convective_time: np.ndarray,
    start: float,
    end: float,
    smoothing: float,
) -> np.ndarray:
    duration = end - start
    if duration <= 0.0 or smoothing <= 0.0:
        raise ValueError("Eldredge duration and smoothing must be positive")
    first = _stable_log_cosh(smoothing * (convective_time - start))
    second = _stable_log_cosh(smoothing * (convective_time - end))
    return (first - second + smoothing * duration) / (2.0 * smoothing * duration)


def mancini_pitch_angle_deg(
    t_star: np.ndarray | float,
    case: Mancini2017Case,
) -> np.ndarray:
    distance = np.asarray(t_star, dtype=float)
    step = _eldredge_step(
        distance,
        0.0,
        case.acceleration_distance_chords,
        case.eldredge_smoothing,
    )
    return (
        case.initial_pitch_deg
        + (case.maximum_pitch_deg - case.initial_pitch_deg) * step
    )


def mancini_periodic_pitch_spacing(
    phase_rad: np.ndarray,
    case: Mancini2017Case,
) -> np.ndarray:
    """Periodic carrier with all closure motion beyond the scored interval."""

    phase = np.mod(np.asarray(phase_rad, dtype=float), 2.0 * np.pi)
    time_star = phase / (2.0 * np.pi) * case.waveform_period_chords
    duration = case.acceleration_distance_chords
    up = _eldredge_step(
        time_star,
        case.warmup_chords,
        case.warmup_chords + duration,
        case.eldredge_smoothing,
    )
    down = _eldredge_step(
        time_star,
        case.closure_start_chords,
        case.closure_start_chords + duration,
        case.eldredge_smoothing,
    )
    restore = _eldredge_step(
        time_star,
        case.closure_start_chords + duration + 1.0,
        case.closure_start_chords + 2.0 * duration + 1.0,
        case.eldredge_smoothing,
    )
    return up - 2.0 * down + restore


def _quality_settings(quality: str) -> tuple[int, int, int]:
    if quality == "smoke":
        return 2, 6, 24
    if quality == "full":
        return 4, 12, 64
    raise ValueError("quality must be smoke or full")


def build_mancini_movement(
    case: Mancini2017Case,
    quality: str = "full",
    *,
    settings: tuple[int, int, int] | None = None,
) -> tuple[Any, dict[str, Any]]:
    nc, ns, steps_per_chord = (
        _quality_settings(quality) if settings is None else settings
    )
    if min(nc, ns, steps_per_chord) <= 0:
        raise ValueError("all Mancini discretization settings must be positive")

    airfoil = ps.geometry.airfoil.Airfoil(name="naca0012")
    root = ps.geometry.wing_cross_section.WingCrossSection(
        airfoil=airfoil,
        chord=case.chord_m,
        num_spanwise_panels=ns,
        spanwise_spacing="cosine",
        control_surface_symmetry_type="symmetric",
    )
    tip = ps.geometry.wing_cross_section.WingCrossSection(
        airfoil=airfoil,
        chord=case.chord_m,
        Lp_Wcsp_Lpp=(0.0, case.semispan_m, 0.0),
        num_spanwise_panels=None,
        spanwise_spacing=None,
        control_surface_symmetry_type="symmetric",
    )
    wing = ps.geometry.wing.Wing(
        name=f"Mancini 2017 AR4 plate {case.case_id}",
        wing_cross_sections=[root, tip],
        symmetric=True,
        symmetryNormal_G=(0.0, 1.0, 0.0),
        symmetryPoint_G_Cg=(0.0, 0.0, 0.0),
        num_chordwise_panels=nc,
        chordwise_spacing="uniform",
    )
    airplane = ps.geometry.airplane.Airplane(
        wings=[wing],
        name=f"Mancini 2017 {case.case_id}",
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

    def pitch_spacing(phase: np.ndarray) -> np.ndarray:
        return mancini_periodic_pitch_spacing(phase, case)

    wing_movement = ps.movements.wing_movement.WingMovement(
        base_wing=wing,
        wing_cross_section_movements=section_movements,
        ampAngles_Gs_to_Wn_ixyz=(0.0, case.maximum_pitch_deg, 0.0),
        periodAngles_Gs_to_Wn_ixyz=(0.0, case.waveform_period_s, 0.0),
        spacingAngles_Gs_to_Wn_ixyz=("sine", pitch_spacing, "sine"),
        rotationPointOffset_Gs_Ler=(0.0, 0.0, 0.0),
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
    delta_time = case.convective_time_s / steps_per_chord
    target_steps = (
        int(np.ceil((case.warmup_chords + case.observation_chords) * steps_per_chord))
        + 1
    )
    movement = ps.movements.movement.Movement(
        airplane_movements=[airplane_movement],
        operating_point_movement=ps.movements.operating_point_movement.OperatingPointMovement(
            base_operating_point=operating_point
        ),
        delta_time=delta_time,
        num_steps=target_steps,
        max_wake_rows=int(np.ceil(6.0 * steps_per_chord)),
    )
    metadata = {
        **case.manifest(),
        "grid_chord_semispan": [nc, ns],
        "steps_per_chord": steps_per_chord,
        "num_steps": movement.num_steps,
        "delta_time_s": movement.delta_time,
        "pitch_trigger_step": int(round(case.warmup_chords * steps_per_chord)),
        "max_wake_rows": int(np.ceil(6.0 * steps_per_chord)),
        "airfoil_adapter": (
            "zero-camber NACA0012 mean surface represents the published 5%-thick "
            "rounded-edge flat plate; thickness and edge radii are unresolved"
        ),
        "span_adapter": (
            "computational midspan symmetry generates the complete AR4 rectangle; "
            "both physical tips remain free in the mirrored full-wing geometry"
        ),
        "motion": (
            f"Mancini Eldredge pitch-up over {case.acceleration_distance_chords:g}c "
            f"with a={case.eldredge_smoothing:g}; periodic closure is outside scoring"
        ),
    }
    return movement, metadata


def run_mancini_fluxv_baseline(
    case: Mancini2017Case,
    quality: str = "full",
    *,
    output_samples: int = 501,
) -> tuple[dict[str, Any], dict[str, Any]]:
    movement, metadata = build_mancini_movement(case, quality)
    problem = ps.problems.UnsteadyProblem(movement=movement, only_final_results=False)
    solver = UVPMHybridSolver(
        problem,
        max_particles=30_000,
        stretch=False,
        free_wake=False,
    )
    solver.run(
        prescribed_wake=True,
        calculate_streamlines=False,
        show_progress=False,
    )

    time_s: list[float] = []
    lift_n: list[float] = []
    drag_n: list[float] = []
    for step in range(solver.unsteady_problem.first_results_step, solver.num_steps):
        force = solver.steady_problems[step].airplanes[0].forces_W
        if force is None:
            continue
        time_s.append(step * movement.delta_time)
        lift_n.append(-float(force[2]))
        drag_n.append(-float(force[0]))
    time = np.asarray(time_s, dtype=float)
    lift = np.asarray(lift_n, dtype=float)
    drag = np.asarray(drag_n, dtype=float)
    t_star = time / case.convective_time_s - case.warmup_chords
    valid = (t_star >= -1.0e-10) & (t_star <= case.observation_chords + 1.0e-10)
    if np.count_nonzero(valid) < 5:
        raise FloatingPointError("Mancini output window is incomplete")
    if not np.all(np.isfinite(lift[valid] + drag[valid])):
        raise FloatingPointError("Mancini solver returned non-finite loads")

    target = np.linspace(0.0, case.observation_chords, output_samples)
    q_area = 0.5 * case.rho_kg_m3 * case.freestream_m_s**2 * case.area_m2
    history: dict[str, Any] = {
        "t_star": target,
        "pitch_deg": mancini_pitch_angle_deg(target, case),
        "lift_n": np.interp(target, t_star[valid], lift[valid]),
        "drag_n": np.interp(target, t_star[valid], drag[valid]),
        "source_step_range": [
            int(np.flatnonzero(valid)[0]),
            int(np.flatnonzero(valid)[-1]),
        ],
        "particle_count": int(solver._vpm_field.np),
        "model_semantics": (
            "unmodified FluxV UVPMHybridSolver load channel: prescribed-wake "
            "finite-wing Ptera UVLM; VPM particles do not feed back into loads"
        ),
    }
    history["CL"] = history["lift_n"] / q_area
    history["CD"] = history["drag_n"] / q_area
    return history, metadata


def apply_frozen_mancini_v4b(
    case: Mancini2017Case,
    baseline_history: dict[str, Any],
    *,
    steps_per_chord: int = 96,
    lesp_critical: float = FROZEN_V4B_LESP_CRITICAL,
) -> dict[str, Any]:
    """Apply the pre-existing transient v4b discrepancy without force fitting."""

    if lesp_critical != FROZEN_V4B_LESP_CRITICAL:
        raise ValueError("Mancini direct transfer keeps the frozen v4b Lcrit=0.11")
    target = np.asarray(baseline_history["t_star"], dtype=float)
    if target.ndim != 1 or target.size < 5 or not np.all(np.isfinite(target)):
        raise ValueError("baseline Mancini t* history is incomplete")
    if steps_per_chord < 8:
        raise ValueError("steps_per_chord must be at least eight")

    delta_time = 1.0 / steps_per_chord
    total_chords = case.warmup_chords + case.observation_chords
    time_star = np.arange(0.0, total_chords + 0.5 * delta_time, delta_time)
    scored_time = time_star - case.warmup_chords
    alpha = np.deg2rad(mancini_pitch_angle_deg(scored_time, case))
    alpha_rate = np.gradient(alpha, delta_time, edge_order=2)
    threshold = LESPThreshold(
        value=FROZEN_V4B_LESP_CRITICAL,
        section_family="5%-thick rounded-edge rectangular flat plate",
        reynolds=case.reynolds,
        source=(
            "unchanged FluxV v4b thin-flat-plate transfer Lcrit=0.11; "
            "not fitted to Mancini Figure 4.13"
        ),
        source_role="pre-existing cross-Re/thickness transfer hypothesis",
    )
    pair = run_ldvm_separation_pair(
        alpha_rad=alpha,
        alpha_rate_per_convective_time=alpha_rate,
        heave_rate_over_u=np.zeros_like(alpha),
        delta_time_convective=delta_time,
        pivot_fraction_chord=case.pivot_fraction_chord,
        threshold=threshold,
        settings=LDVMSectionSettings(
            ndiv=50,
            naterm=24,
            core_radius_time_step_ratio=1.3,
            max_wake_steps=time_star.size,
        ),
    )
    valid = (scored_time >= -1.0e-10) & (
        scored_time <= case.observation_chords + 1.0e-10
    )
    angle = np.interp(target, scored_time[valid], alpha[valid])

    def sample(name: str) -> np.ndarray:
        return np.interp(
            target,
            scored_time[valid],
            np.asarray(pair["delta"][name], dtype=float)[valid],
        )

    projection = project_ldvm_delta_to_finite_wing(
        sample("CNc"),
        sample("CNnc"),
        sample("CNnonl"),
        sample("CSf"),
        angle,
        aspect_ratio=case.aspect_ratio,
    )
    delta_cl = np.asarray(projection["delta_CL"], dtype=float)
    delta_cd = np.asarray(projection["delta_CD"], dtype=float)
    baseline_cl = np.asarray(baseline_history["CL"], dtype=float)
    baseline_cd = np.asarray(baseline_history["CD"], dtype=float)
    if baseline_cl.shape != target.shape or baseline_cd.shape != target.shape:
        raise ValueError("baseline Mancini coefficient histories are not aligned")
    corrected_cl = baseline_cl + delta_cl
    corrected_cd = baseline_cd + delta_cd
    if not np.all(np.isfinite(corrected_cl + corrected_cd)):
        raise FloatingPointError("Mancini v4b returned non-finite loads")
    q_area = 0.5 * case.rho_kg_m3 * case.freestream_m_s**2 * case.area_m2
    return {
        "t_star": target.copy(),
        "pitch_deg": np.asarray(baseline_history["pitch_deg"], dtype=float).copy(),
        "CL": corrected_cl,
        "CD": corrected_cd,
        "lift_n": corrected_cl * q_area,
        "drag_n": corrected_cd * q_area,
        "delta_CL": delta_cl,
        "delta_CD": delta_cd,
        "A0": np.interp(
            target,
            scored_time[valid],
            np.asarray(pair["separated"]["A0"], dtype=float)[valid],
        ),
        "shed_lev": np.interp(
            target,
            scored_time[valid],
            np.asarray(pair["shed_lev"], dtype=float)[valid],
        )
        >= 0.5,
        "lesp_critical": FROZEN_V4B_LESP_CRITICAL,
        "lesp_provenance": threshold.manifest(),
        "finite_wing_normal_gain": projection["normal_gain"],
        "finite_wing_added_mass_gain": projection["added_mass_gain"],
        "finite_wing_nonlinear_normal_gain": projection["nonlinear_normal_gain"],
        "finite_wing_axial_suction_gain": projection["axial_suction_gain"],
        "steps_per_chord": int(steps_per_chord),
        "model_semantics": (
            "FluxV v4b direct transfer: retained finite-wing UVLM plus causal "
            "clean-room Ramesh LDVM separated-minus-attached increment; no "
            "Mancini force observation enters the prediction"
        ),
        "limitations": [
            "Lcrit=0.11 is transferred across Reynolds number and plate thickness",
            "LDVM increment is a 2-D section discrepancy, not a common 3-D LEV wake",
            "UVLM mean surface omits 5% thickness and rounded edge radii",
            "experiment publishes whole-wing lift, not spanwise sectional loads",
        ],
    }


__all__ = [
    "FROZEN_V4B_LESP_CRITICAL",
    "MANCINI_2017_CASES",
    "MANCINI_EXPERIMENT_HAS_SPANWISE_LOADS",
    "Mancini2017Case",
    "apply_frozen_mancini_v4b",
    "build_mancini_movement",
    "load_mancini_fig4_13b_experiment",
    "mancini_pitch_angle_deg",
    "mancini_periodic_pitch_spacing",
    "run_mancini_fluxv_baseline",
]
