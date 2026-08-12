"""Stevens & Babinsky (2017) finite-wing pitching experiment.

The experiment uses a rectangular carbon plate translated at constant speed
and pitched from 0 to 45 degrees over one chord of travel.  A skim plate acts
as the symmetry plane, producing an effective aspect ratio of four.  This
module reconstructs the two reported pitch axes without using the measured
lift histories as model inputs.

Only lift was published.  A drag prediction can be exported as a diagnostic,
but there is no experimental drag target for this paper.
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


REPO_ROOT = Path(__file__).resolve().parents[2]
STEVENS_FIG21_EXPERIMENT_CSV = (
    REPO_ROOT / "docs/forward_flight_large_pitch/reproductions/"
    "unified_fluxv_v4_ldvm_stevens_20260812/source_data/"
    "stevens2017_fig21_experiment_digitized.csv"
)
STEVENS_FIG21_EXPERIMENT_SHA256 = (
    "793c3c2af626de1f8e46e788e814f4735c903d3dfb198cd5e5257964fdac51fa"
)
STEVENS_EXPERIMENTAL_OBSERVABLES = ("CL",)
STEVENS_EXPERIMENT_HAS_DRAG = False


@dataclass(frozen=True)
class Stevens2017Case:
    case_id: str = "stevens2017_pitching_flat_plate"
    chord_m: float = 0.120
    effective_aspect_ratio: float = 4.0
    thickness_fraction: float = 0.025
    reynolds: float = 10_000.0
    freestream_m_s: float = 0.080
    rho_kg_m3: float = 998.0
    maximum_pitch_deg: float = 45.0
    reduced_pitch_rate: float = float(np.pi / 8.0)
    eldredge_smoothing: float = 11.0
    warmup_chords: float = 2.0
    observation_chords: float = 5.0
    waveform_period_chords: float = 12.0

    @property
    def span_m(self) -> float:
        return self.effective_aspect_ratio * self.chord_m

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
    def pitch_rate_rad_s(self) -> float:
        return 2.0 * self.reduced_pitch_rate * self.freestream_m_s / self.chord_m

    @property
    def waveform_period_s(self) -> float:
        return self.waveform_period_chords * self.convective_time_s

    def manifest(self) -> dict[str, float | str]:
        output = asdict(self)
        output.update(
            span_m=self.span_m,
            semispan_m=self.semispan_m,
            area_m2=self.area_m2,
            nu_m2_s=self.nu_m2_s,
            convective_time_s=self.convective_time_s,
            pitch_rate_rad_s=self.pitch_rate_rad_s,
            waveform_period_s=self.waveform_period_s,
            dimensional_speed_provenance=(
                "U=0.080 m/s is the similarity speed implied by the reported "
                "wind-off alpha_dot=0.523 rad/s and k=0.392; the inviscid "
                "coefficient prediction depends on k, not on this scale choice."
            ),
        )
        return output


STEVENS_2017 = Stevens2017Case()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_stevens_fig21_experiment(
    path: Path = STEVENS_FIG21_EXPERIMENT_CSV,
    *,
    verify_hash: bool = True,
) -> dict[str, Any]:
    """Load the two published Figure-21 experimental lift histories.

    The contract is intentionally lift-only.  Stevens & Babinsky used a
    two-component balance but explicitly reported only lift, so neither this
    loader nor the source CSV synthesizes an experimental drag channel.
    """

    digest = _sha256(path)
    if verify_hash and digest != STEVENS_FIG21_EXPERIMENT_SHA256:
        raise ValueError(
            "Stevens Figure-21 CSV hash mismatch: "
            f"expected {STEVENS_FIG21_EXPERIMENT_SHA256}, got {digest}"
        )
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        expected_fields = (
            "s_over_c",
            "experiment_CL_leading_edge_axis",
            "experiment_CL_mid_chord_axis",
            "source_figure",
        )
        if tuple(reader.fieldnames or ()) != expected_fields:
            raise ValueError(
                "unexpected Stevens Figure-21 columns: " f"{reader.fieldnames!r}"
            )
        rows = list(reader)
    if len(rows) != 501:
        raise ValueError(f"expected 501 Figure-21 samples, got {len(rows)}")

    distance = np.asarray([float(row["s_over_c"]) for row in rows])
    leading_edge = np.asarray(
        [float(row["experiment_CL_leading_edge_axis"]) for row in rows]
    )
    mid_chord = np.asarray([float(row["experiment_CL_mid_chord_axis"]) for row in rows])
    source_figure = np.asarray([int(row["source_figure"]) for row in rows], dtype=int)
    if not np.allclose(distance, np.linspace(0.0, 5.0, 501), atol=1.0e-12):
        raise ValueError("Figure-21 distance grid is not the frozen 0.01c grid")
    if not np.all(source_figure == 21):
        raise ValueError("Stevens experimental rows must all cite Figure 21")
    if not np.all(np.isfinite(leading_edge + mid_chord)):
        raise ValueError("Stevens Figure-21 lift history contains non-finite data")

    return {
        "s_over_c": distance,
        "CL_leading_edge_axis": leading_edge,
        "CL_mid_chord_axis": mid_chord,
        "source_figure": source_figure,
        "observed_channels": STEVENS_EXPERIMENTAL_OBSERVABLES,
        "drag_available": STEVENS_EXPERIMENT_HAS_DRAG,
        "digitization_sha256": digest,
        "data_role": "digitized_experimental_lift_ground_truth",
    }


def _stable_log_cosh(value: np.ndarray) -> np.ndarray:
    return np.logaddexp(value, -value) - np.log(2.0)


def _eldredge_step(
    convective_time: np.ndarray,
    start: float,
    end: float,
    smoothing: float,
) -> np.ndarray:
    """Unit Wang--Eldredge smoothed step from zero to one."""

    duration = end - start
    if duration <= 0.0:
        raise ValueError("Eldredge step duration must be positive")
    first = _stable_log_cosh(smoothing * (convective_time - start))
    second = _stable_log_cosh(smoothing * (convective_time - end))
    return (first - second + smoothing * duration) / (2.0 * smoothing * duration)


def stevens_periodic_pitch_spacing(
    phase_rad: np.ndarray,
    case: Stevens2017Case = STEVENS_2017,
) -> np.ndarray:
    """Periodic carrier whose first up-ramp is the published 0-to-45 motion.

    Ptera custom motions must be periodic and have unit amplitude.  The
    validation window ends at ``s/c=5``.  A distant down-ramp and recovery are
    placed outside that window so the carrier closes without altering the
    experiment being scored.
    """

    phase = np.mod(np.asarray(phase_rad, dtype=float), 2.0 * np.pi)
    convective_time = phase / (2.0 * np.pi) * case.waveform_period_chords
    up = _eldredge_step(
        convective_time,
        case.warmup_chords,
        case.warmup_chords + 1.0,
        case.eldredge_smoothing,
    )
    down_to_negative = _eldredge_step(
        convective_time, 8.0, 9.0, case.eldredge_smoothing
    )
    return_to_zero = _eldredge_step(
        convective_time, 10.0, 11.0, case.eldredge_smoothing
    )
    return up - 2.0 * down_to_negative + return_to_zero


def stevens_pitch_angle_deg(
    s_over_c: np.ndarray | float,
    case: Stevens2017Case = STEVENS_2017,
) -> np.ndarray:
    """Published pitch-up history with ``s/c=0`` at the trigger."""

    distance = np.asarray(s_over_c, dtype=float)
    return case.maximum_pitch_deg * _eldredge_step(
        distance,
        0.0,
        1.0,
        case.eldredge_smoothing,
    )


def _quality_settings(quality: str) -> tuple[int, int, int]:
    if quality == "smoke":
        return 2, 6, 24
    if quality == "full":
        return 4, 12, 64
    raise ValueError("quality must be 'smoke' or 'full'")


def build_stevens_movement(
    pivot_fraction_chord: float,
    quality: str = "full",
    case: Stevens2017Case = STEVENS_2017,
    settings: tuple[int, int, int] | None = None,
) -> tuple[Any, dict[str, Any]]:
    if not 0.0 <= pivot_fraction_chord <= 1.0:
        raise ValueError("pitch pivot must lie on the chord")
    nc, ns, steps_per_chord = (
        _quality_settings(quality) if settings is None else settings
    )
    if min(nc, ns, steps_per_chord) <= 0:
        raise ValueError("all discretization settings must be positive")

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
    pivot_m = pivot_fraction_chord * case.chord_m
    wing = ps.geometry.wing.Wing(
        name=("Stevens 2017 effective AR4 plate " f"pivot={pivot_fraction_chord:g}c"),
        wing_cross_sections=[root, tip],
        Ler_Gs_Cgs=(-pivot_m, 0.0, 0.0),
        symmetric=True,
        symmetryNormal_G=(0.0, 1.0, 0.0),
        symmetryPoint_G_Cg=(0.0, 0.0, 0.0),
        num_chordwise_panels=nc,
        chordwise_spacing="uniform",
    )
    airplane = ps.geometry.airplane.Airplane(
        wings=[wing],
        name=f"Stevens 2017 pivot {pivot_fraction_chord:g}c",
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
        ampAngles_Gs_to_Wn_ixyz=(0.0, case.maximum_pitch_deg, 0.0),
        periodAngles_Gs_to_Wn_ixyz=(0.0, case.waveform_period_s, 0.0),
        spacingAngles_Gs_to_Wn_ixyz=(
            "sine",
            stevens_periodic_pitch_spacing,
            "sine",
        ),
        rotationPointOffset_Gs_Ler=(pivot_m, 0.0, 0.0),
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
        "case_id": case.case_id,
        "pivot_fraction_chord": float(pivot_fraction_chord),
        "grid_chord_semispan": [nc, ns],
        "steps_per_chord": steps_per_chord,
        "num_steps": movement.num_steps,
        "delta_time_s": movement.delta_time,
        "pitch_trigger_step": int(round(case.warmup_chords * steps_per_chord)),
        "max_wake_rows": int(np.ceil(6.0 * steps_per_chord)),
        "airfoil_adapter": (
            "zero-camber NACA0012 mean surface for the 2.5%-thick rounded-LE "
            "carbon plate; thickness and exact LE radius are not represented"
        ),
        "symmetry_adapter": (
            "Ptera symmetric full wing represents the skim-plate mirror of the "
            "physical half-wing; reference area is the effective full-wing area"
        ),
        "motion": (
            "Wang-Eldredge a_s=11 pitch-up over one convected chord; distant "
            "periodic closure lies outside the scored 0<=s/c<=5 window"
        ),
    }
    return movement, metadata


def run_stevens_fluxv_baseline(
    pivot_fraction_chord: float,
    quality: str = "full",
    case: Stevens2017Case = STEVENS_2017,
    output_samples: int = 501,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the untouched FluxV UVLM load channel for one pitch axis."""

    movement, metadata = build_stevens_movement(
        pivot_fraction_chord=pivot_fraction_chord,
        quality=quality,
        case=case,
    )
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
    time_s_array = np.asarray(time_s, dtype=float)
    lift_array = np.asarray(lift_n, dtype=float)
    drag_array = np.asarray(drag_n, dtype=float)
    s_over_c = time_s_array / case.convective_time_s - case.warmup_chords
    valid = (s_over_c >= -1.0e-10) & (s_over_c <= case.observation_chords + 1.0e-10)
    if np.count_nonzero(valid) < 5:
        raise FloatingPointError("Stevens output window is incomplete")
    if not np.all(np.isfinite(lift_array[valid] + drag_array[valid])):
        raise FloatingPointError("Stevens solver returned non-finite loads")

    target = np.linspace(0.0, case.observation_chords, output_samples)
    q_area = 0.5 * case.rho_kg_m3 * case.freestream_m_s**2 * case.area_m2
    history = {
        "s_over_c": target,
        "pitch_deg": stevens_pitch_angle_deg(target, case),
        "lift_n": np.interp(target, s_over_c[valid], lift_array[valid]),
        "drag_n": np.interp(target, s_over_c[valid], drag_array[valid]),
    }
    history["CL"] = history["lift_n"] / q_area
    history["CD"] = history["drag_n"] / q_area
    history["model_semantics"] = (
        "unmodified FluxV UVPMHybridSolver load channel: prescribed-wake "
        "Ptera UVLM; VPM particles are one-way and do not feed back into loads"
    )
    history["source_step_range"] = [
        int(np.flatnonzero(valid)[0]),
        int(np.flatnonzero(valid)[-1]),
    ]
    history["particle_count"] = int(solver._vpm_field.np)
    return history, metadata


def run_stevens_ldvm_strips(
    pivot_fraction_chord: float,
    *,
    lesp_critical: float,
    steps_per_chord: int = 96,
    case: Stevens2017Case = STEVENS_2017,
    output_samples: int = 501,
) -> dict[str, Any]:
    """Run the clean-room Ramesh LDVM as finite-wing strip sections.

    This is an explicit diagnostic bridge, not a three-dimensional LDVM.  Each
    span station runs the published two-dimensional section dynamics; a
    Prandtl slope ratio supplies the only finite-wing scaling.  The function
    requires an explicit ``lesp_critical`` because Ramesh's threshold is
    section/Re dependent and cannot be silently transferred from SD7003.
    """

    if lesp_critical <= 0.0:
        raise ValueError("LESP critical value must be explicitly positive")
    if steps_per_chord < 8:
        raise ValueError("steps_per_chord must be at least eight")
    # Import lazily so users interested only in the UVLM benchmark do not pull
    # the legacy diagnostic module into normal package import.
    from ldvm_fourier import LDVM2D

    dt_star = 1.0 / steps_per_chord
    model = LDVM2D(
        U=1.0,
        c=1.0,
        ndiv=70,
        naterm=35,
        dt=dt_star,
        rho=1.0,
        lesp_crit=lesp_critical,
        camber_m=0.0,
        pivot_xc=pivot_fraction_chord,
        core_rc=1.3 * dt_star,
        max_wake=int(12 * steps_per_chord),
    )
    total_chords = case.warmup_chords + case.observation_chords
    time_star = np.arange(0.0, total_chords + 0.5 * dt_star, dt_star)
    scored_distance = time_star - case.warmup_chords
    alpha = np.deg2rad(stevens_pitch_angle_deg(scored_distance, case))
    # The smoothed law is evaluated analytically by a centred derivative; its
    # endpoints lie in long, nominally constant segments.
    alpha_dot = np.gradient(alpha, dt_star, edge_order=2)
    lift = np.empty(time_star.size, dtype=float)
    drag = np.empty(time_star.size, dtype=float)
    a0 = np.empty(time_star.size, dtype=float)
    shed_lev = np.empty(time_star.size, dtype=bool)
    model_history_lev_count = 0
    for index, (angle, rate) in enumerate(zip(alpha, alpha_dot)):
        result = model.step(float(angle), float(rate), 0.0)
        lift[index] = result["CLf"]
        drag[index] = result["CDf"]
        a0[index] = result["A0"]
        shed_lev[index] = result["n_lev"] > model_history_lev_count
        model_history_lev_count = int(result["n_lev"])

    valid = (scored_distance >= -1.0e-10) & (
        scored_distance <= case.observation_chords + 1.0e-10
    )
    target = np.linspace(0.0, case.observation_chords, output_samples)
    finite_wing_slope = 2.0 * np.pi / (1.0 + 2.0 / case.effective_aspect_ratio)
    finite_wing_gain = finite_wing_slope / (2.0 * np.pi)
    return {
        "s_over_c": target,
        "pitch_deg": stevens_pitch_angle_deg(target, case),
        "CL": finite_wing_gain * np.interp(target, scored_distance[valid], lift[valid]),
        "CD": finite_wing_gain * np.interp(target, scored_distance[valid], drag[valid]),
        "A0": np.interp(target, scored_distance[valid], a0[valid]),
        "shed_lev": np.interp(
            target,
            scored_distance[valid],
            shed_lev[valid].astype(float),
        )
        >= 0.5,
        "lesp_critical": float(lesp_critical),
        "finite_wing_gain": float(finite_wing_gain),
        "steps_per_chord": int(steps_per_chord),
        "model_semantics": (
            "clean-room Ramesh LDVM v2.5 two-dimensional section dynamics, "
            "integrated as independent strips with a Prandtl slope ratio; this "
            "is not a three-dimensional LDVM or the final UVLM-coupled model"
        ),
    }


def run_stevens_fluxv_ldvm_increment(
    pivot_fraction_chord: float,
    baseline_history: dict[str, Any],
    *,
    lesp_critical: float,
    threshold_source: str,
    steps_per_chord: int = 96,
    case: Stevens2017Case = STEVENS_2017,
) -> dict[str, Any]:
    """Add a separated-minus-attached LDVM increment to retained UVLM loads.

    The normal-force increment is scaled by the finite-wing lift-slope ratio
    ``g``.  Ramesh's axial-suction term is quadratic in LESP/circulation, so it
    is scaled by ``g**2``.  No measured Stevens force enters this calculation.
    """

    from forward_flight_benchmarks.ldvm_uvlm_correction import (
        LDVMSectionSettings,
        LESPThreshold,
        project_ldvm_delta_to_finite_wing,
        run_ldvm_separation_pair,
    )

    target = np.asarray(baseline_history["s_over_c"], dtype=float)
    if target.ndim != 1 or target.size < 5 or not np.all(np.isfinite(target)):
        raise ValueError("baseline s/c history is incomplete")
    if steps_per_chord < 8:
        raise ValueError("steps_per_chord must be at least eight")

    delta_time = 1.0 / steps_per_chord
    total_chords = case.warmup_chords + case.observation_chords
    time_star = np.arange(0.0, total_chords + 0.5 * delta_time, delta_time)
    scored_distance = time_star - case.warmup_chords
    alpha = np.deg2rad(stevens_pitch_angle_deg(scored_distance, case))
    alpha_rate = np.gradient(alpha, delta_time, edge_order=2)
    threshold = LESPThreshold(
        value=lesp_critical,
        section_family="thin rounded leading-edge plate",
        reynolds=case.reynolds,
        source=threshold_source,
    )
    pair = run_ldvm_separation_pair(
        alpha_rad=alpha,
        alpha_rate_per_convective_time=alpha_rate,
        heave_rate_over_u=np.zeros_like(alpha),
        delta_time_convective=delta_time,
        pivot_fraction_chord=pivot_fraction_chord,
        threshold=threshold,
        settings=LDVMSectionSettings(
            ndiv=50,
            naterm=24,
            core_radius_time_step_ratio=1.3,
            max_wake_steps=time_star.size,
        ),
    )
    valid = (scored_distance >= -1.0e-10) & (
        scored_distance <= case.observation_chords + 1.0e-10
    )
    angle = np.interp(target, scored_distance[valid], alpha[valid])
    delta_cnc = np.interp(
        target,
        scored_distance[valid],
        np.asarray(pair["delta"]["CNc"])[valid],
    )
    delta_cnnc = np.interp(
        target,
        scored_distance[valid],
        np.asarray(pair["delta"]["CNnc"])[valid],
    )
    delta_cn_nonl = np.interp(
        target,
        scored_distance[valid],
        np.asarray(pair["delta"]["CNnonl"])[valid],
    )
    delta_cs = np.interp(
        target,
        scored_distance[valid],
        np.asarray(pair["delta"]["CSf"])[valid],
    )
    projection = project_ldvm_delta_to_finite_wing(
        delta_cnc,
        delta_cnnc,
        delta_cn_nonl,
        delta_cs,
        angle,
        aspect_ratio=case.effective_aspect_ratio,
    )
    delta_cl = np.asarray(projection["delta_CL"], dtype=float)
    delta_cd = np.asarray(projection["delta_CD"], dtype=float)
    baseline_cl = np.asarray(baseline_history["CL"], dtype=float)
    baseline_cd = np.asarray(baseline_history["CD"], dtype=float)
    if baseline_cl.shape != target.shape or baseline_cd.shape != target.shape:
        raise ValueError("baseline coefficient histories are not aligned")
    q_area = 0.5 * case.rho_kg_m3 * case.freestream_m_s**2 * case.area_m2
    corrected_cl = baseline_cl + delta_cl
    corrected_cd = baseline_cd + delta_cd
    return {
        "s_over_c": target.copy(),
        "pitch_deg": np.asarray(baseline_history["pitch_deg"], dtype=float).copy(),
        "CL": corrected_cl,
        "CD": corrected_cd,
        "lift_n": corrected_cl * q_area,
        "drag_n": corrected_cd * q_area,
        "delta_CL": delta_cl,
        "delta_CD": delta_cd,
        "A0": np.interp(
            target,
            scored_distance[valid],
            np.asarray(pair["separated"]["A0"])[valid],
        ),
        "shed_lev": np.interp(
            target,
            scored_distance[valid],
            np.asarray(pair["shed_lev"], dtype=float)[valid],
        )
        >= 0.5,
        "lesp_critical": float(lesp_critical),
        "threshold_source": threshold_source,
        "finite_wing_normal_gain": projection["normal_gain"],
        "finite_wing_added_mass_gain": projection["added_mass_gain"],
        "finite_wing_nonlinear_normal_gain": projection["nonlinear_normal_gain"],
        "finite_wing_axial_suction_gain": projection["axial_suction_gain"],
        "steps_per_chord": int(steps_per_chord),
        "model_semantics": (
            "FluxV v4 diagnostic: retained finite-wing UVLM plus causal "
            "clean-room Ramesh LDVM separated-minus-attached increment; "
            "normal force scales with g and quadratic axial suction with g^2"
        ),
    }
