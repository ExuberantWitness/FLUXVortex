"""PteraSoftware/FluxV adapters for the reconstructed rigid-wing cases."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pterasoftware as ps

from fluxvortex.solver import UVPMHybridSolver

from .cases import (
    IZRAELEVITZ_2017,
    IZRAELEVITZ_2017_FIG11,
    IZRAELEVITZ_2017_FIG14_SCHERER,
    YANG_2023,
    YANG_2025,
    IzraelevitzCase,
    IzraelevitzHeavePitchCase,
    IzraelevitzSchererCase,
    YangCase,
    Yang2025RigidCase,
    fourbar_extrema_deg,
    izraelevitz_euler_spacing,
    yang2025_fourbar_spacing,
    yang_fourbar_spacing,
)


MODEL_SEMANTICS = {
    "fluxv_uvpm": (
        "Actual fluxvortex.solver.UVPMHybridSolver; aerodynamic loads are the "
        "parent prescribed-ring UVLM channel. VPM particles are shed and "
        "advected one-way and do not feed back into the wing loads."
    ),
    "ptera_prescribed_wake_uvlm": (
        "PteraSoftware ring-vortex UVLM with prescribed wake; load-equivalent "
        "control for the present FluxV implementation."
    ),
    "ptera_free_wake_uvlm": (
        "PteraSoftware ring-vortex UVLM with a free-convected ring wake."
    ),
}


def _quality_settings(quality: str, case_name: str) -> tuple[int, int, int, int]:
    """Return chord panels, semispan panels, steps/cycle, cycles."""

    if quality == "smoke":
        return (2, 4, 20, 2) if case_name == "yang" else (2, 6, 24, 2)
    return (4, 8, 64, 4) if case_name == "yang" else (4, 10, 72, 4)


def build_yang_movement(
    aoa_deg: float,
    quality: str = "full",
    case: YangCase = YANG_2023,
) -> tuple[Any, dict[str, Any]]:
    nc, ns, steps_per_cycle, cycles = _quality_settings(quality, "yang")
    airfoil = ps.geometry.airfoil.Airfoil(name="naca0012")
    sections = [
        ps.geometry.wing_cross_section.WingCrossSection(
            airfoil=airfoil,
            chord=case.chord_m,
            Lp_Wcsp_Lpp=(0.0, 0.0, 0.0),
            num_spanwise_panels=ns,
            spanwise_spacing="uniform",
        ),
        ps.geometry.wing_cross_section.WingCrossSection(
            airfoil=airfoil,
            chord=case.chord_m,
            Lp_Wcsp_Lpp=(0.0, case.span_m, 0.0),
            num_spanwise_panels=None,
            spanwise_spacing=None,
        ),
    ]
    flap_min, flap_max = fourbar_extrema_deg(case)
    flap_mean = 0.5 * (flap_min + flap_max)
    flap_amplitude = 0.5 * (flap_max - flap_min)
    wing = ps.geometry.wing.Wing(
        name="Yang 2023 rigid rectangular single wing",
        wing_cross_sections=sections,
        angles_Gs_to_Wn_ixyz=(flap_mean, float(aoa_deg), 0.0),
        symmetric=False,
        num_chordwise_panels=nc,
        chordwise_spacing="uniform",
    )
    airplane = ps.geometry.airplane.Airplane(
        wings=[wing],
        name=f"Yang 2023 rigid AoA {aoa_deg:g}",
        s_ref=case.area_m2,
        c_ref=case.chord_m,
        b_ref=case.span_m,
    )
    section_movements = [
        ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
            base_wing_cross_section=section
        )
        for section in sections
    ]
    wing_movement = ps.movements.wing_movement.WingMovement(
        base_wing=wing,
        wing_cross_section_movements=section_movements,
        ampAngles_Gs_to_Wn_ixyz=(flap_amplitude, 0.0, 0.0),
        periodAngles_Gs_to_Wn_ixyz=(case.period_s, 0.0, 0.0),
        spacingAngles_Gs_to_Wn_ixyz=(yang_fourbar_spacing, "sine", "sine"),
        phaseAngles_Gs_to_Wn_ixyz=(0.0, 0.0, 0.0),
    )
    airplane_movement = ps.movements.airplane_movement.AirplaneMovement(
        base_airplane=airplane, wing_movements=[wing_movement]
    )
    operating_point = ps.operating_point.OperatingPoint(
        rho=case.rho_kg_m3,
        vCg__E=case.freestream_m_s,
        alpha=0.0,
        beta=0.0,
        nu=case.nu_m2_s,
    )
    requested_delta_time = (
        None if steps_per_cycle == 0 else case.period_s / steps_per_cycle
    )
    movement = ps.movements.movement.Movement(
        airplane_movements=[airplane_movement],
        operating_point_movement=ps.movements.operating_point_movement.OperatingPointMovement(
            base_operating_point=operating_point
        ),
        delta_time=requested_delta_time,
        num_cycles=cycles,
        max_wake_cycles=min(2, cycles),
    )
    metadata = {
        "grid_chord_span": [nc, ns],
        "steps_per_cycle": case.period_s / movement.delta_time,
        "requested_steps_per_cycle": steps_per_cycle,
        "cycles": cycles,
        "aoa_deg": float(aoa_deg),
        "flap_mean_deg": flap_mean,
        "flap_amplitude_deg": flap_amplitude,
        "airfoil_adapter": "zero-camber NACA0012 mean surface for 1 mm balsa plate",
    }
    return movement, metadata


def build_yang2025_movement(
    aoa_deg: float,
    quality: str = "full",
    case: Yang2025RigidCase = YANG_2025,
    settings: tuple[int, int, int, int, int] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Build the formal-paper rigid wing with nominal four-bar kinematics.

    Yang et al. drove their numerical validation with a laser-measured wing
    angle history that is not public.  This builder therefore uses the rounded
    link lengths published in the paper and records that distinction in every
    returned manifest.  The aerodynamic wing begins ``wing_root_offset_m``
    from the flapping joint and is rotated about the joint, not about its own
    inboard aerodynamic edge.
    """

    if settings is None:
        if quality == "smoke":
            nc, ns, steps_per_cycle, cycles, wake_cycles = (2, 4, 20, 2, 2)
        else:
            nc, ns, steps_per_cycle, cycles, wake_cycles = (
                case.chordwise_panels,
                case.spanwise_panels,
                case.steps_per_cycle,
                3,
                2,
            )
    else:
        nc, ns, steps_per_cycle, cycles, wake_cycles = settings
    if min(nc, ns, steps_per_cycle, cycles, wake_cycles) <= 0:
        raise ValueError("all Yang 2025 discretization settings must be positive")
    if wake_cycles > cycles:
        raise ValueError("max_wake_cycles cannot exceed num_cycles")

    airfoil = ps.geometry.airfoil.Airfoil(name="naca0012")
    sections = [
        ps.geometry.wing_cross_section.WingCrossSection(
            airfoil=airfoil,
            chord=case.chord_m,
            Lp_Wcsp_Lpp=(0.0, 0.0, 0.0),
            num_spanwise_panels=ns,
            spanwise_spacing="uniform",
        ),
        ps.geometry.wing_cross_section.WingCrossSection(
            airfoil=airfoil,
            chord=case.chord_m,
            Lp_Wcsp_Lpp=(0.0, case.span_m, 0.0),
            num_spanwise_panels=None,
            spanwise_spacing=None,
        ),
    ]
    flap_min, flap_max = fourbar_extrema_deg(case)
    flap_mean = 0.5 * (flap_min + flap_max)
    flap_amplitude = 0.5 * (flap_max - flap_min)
    wing = ps.geometry.wing.Wing(
        name="Yang 2025 rigid rectangular single wing",
        wing_cross_sections=sections,
        Ler_Gs_Cgs=(0.0, case.wing_root_offset_m, 0.0),
        angles_Gs_to_Wn_ixyz=(flap_mean, float(aoa_deg), 0.0),
        symmetric=False,
        num_chordwise_panels=nc,
        chordwise_spacing="uniform",
    )
    airplane = ps.geometry.airplane.Airplane(
        wings=[wing],
        name=f"Yang 2025 rigid AoA {aoa_deg:g}",
        s_ref=case.area_m2,
        c_ref=case.chord_m,
        b_ref=case.span_m,
    )
    section_movements = [
        ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
            base_wing_cross_section=section
        )
        for section in sections
    ]
    wing_movement = ps.movements.wing_movement.WingMovement(
        base_wing=wing,
        wing_cross_section_movements=section_movements,
        ampAngles_Gs_to_Wn_ixyz=(flap_amplitude, 0.0, 0.0),
        periodAngles_Gs_to_Wn_ixyz=(case.period_s, 0.0, 0.0),
        spacingAngles_Gs_to_Wn_ixyz=(yang2025_fourbar_spacing, "sine", "sine"),
        phaseAngles_Gs_to_Wn_ixyz=(0.0, 0.0, 0.0),
        rotationPointOffset_Gs_Ler=(0.0, -case.wing_root_offset_m, 0.0),
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
        "case_id": case.case_id,
        "grid_chord_span": [nc, ns],
        "steps_per_cycle": case.period_s / movement.delta_time,
        "requested_steps_per_cycle": steps_per_cycle,
        "cycles": cycles,
        "max_wake_cycles": wake_cycles,
        "aoa_deg": float(aoa_deg),
        "flap_min_deg": flap_min,
        "flap_max_deg": flap_max,
        "flap_mean_deg": flap_mean,
        "flap_amplitude_deg": flap_amplitude,
        "wing_root_offset_m": case.wing_root_offset_m,
        "rotation_point_offset_Gs_Ler_m": [0.0, -case.wing_root_offset_m, 0.0],
        "kinematics": "nominal-fourbar-from-rounded-JFS-2025-links",
        "paper_input_missing": "per-case laser-displacement wing-angle history",
        "airfoil_adapter": "zero-camber NACA0012 mean surface for 1 mm balsa plate",
    }
    return movement, metadata


def build_izraelevitz_movement(
    quality: str = "full",
    case: IzraelevitzCase = IZRAELEVITZ_2017,
    settings: tuple[int, int, int, int, int] | None = None,
) -> tuple[Any, dict[str, Any]]:
    if settings is None:
        nc, ns, steps_per_cycle, cycles = _quality_settings(quality, "izraelevitz")
        wake_cycles = min(2, cycles)
    else:
        nc, ns, steps_per_cycle, cycles, wake_cycles = settings
    airfoil = ps.geometry.airfoil.Airfoil(name="naca0012")
    # Cosine-spaced blade boundaries, matching the paper's n=10 full-case
    # discretization.  A zero tip chord is singular for a panel solver, so only
    # the final node is regularized to 0.2% c_mid.
    eta = np.sin(np.linspace(0.0, 0.5 * np.pi, ns + 1))
    span_nodes = case.semispan_m * eta
    chord_nodes = case.chord_m(span_nodes)
    tip_chord = 0.002 * case.midspan_chord_m
    chord_nodes[-1] = tip_chord
    wings = []
    all_section_movements = []
    for side_name, mirror_only in (("right", False), ("left", True)):
        sections = []
        section_movements = []
        for index, (span, chord) in enumerate(zip(span_nodes, chord_nodes)):
            offset = 0.0 if index == 0 else span - span_nodes[index - 1]
            section = ps.geometry.wing_cross_section.WingCrossSection(
                airfoil=airfoil,
                chord=float(chord),
                Lp_Wcsp_Lpp=(0.0, float(offset), 0.0),
                num_spanwise_panels=1 if index < ns else None,
                spanwise_spacing="uniform" if index < ns else None,
            )
            twist_amplitude = abs(case.tip_twist_amplitude_deg) * float(eta[index])
            sections.append(section)
            section_movements.append(
                ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
                    base_wing_cross_section=section,
                    ampAngles_Wcsp_to_Wcs_ixyz=(0.0, twist_amplitude, 0.0),
                    periodAngles_Wcsp_to_Wcs_ixyz=(
                        0.0,
                        case.period_s if twist_amplitude else 0.0,
                        0.0,
                    ),
                    phaseAngles_Wcsp_to_Wcs_ixyz=(
                        0.0,
                        180.0 if twist_amplitude else 0.0,
                        0.0,
                    ),
                )
            )
        wings.append(
            ps.geometry.wing.Wing(
                name=f"Izraelevitz 2017 elliptic AR6 {side_name} half-wing",
                wing_cross_sections=sections,
                symmetric=False,
                mirror_only=mirror_only,
                symmetryNormal_G=(0.0, 1.0, 0.0) if mirror_only else None,
                # A 0.1 mm plane offset prevents Ptera from switching its internal
                # mirror symmetry type exactly when the moving wing passes through
                # the global x-z plane.  Its geometric effect is negligible and is
                # recorded in the run metadata below.
                symmetryPoint_G_Cg=(0.0, 1.0e-4, 0.0) if mirror_only else None,
                num_chordwise_panels=nc,
                chordwise_spacing="uniform",
            )
        )
        all_section_movements.append(section_movements)
    airplane = ps.geometry.airplane.Airplane(
        wings=wings,
        name="Izraelevitz 2017 Figure 13",
        s_ref=case.area_m2,
        c_ref=case.midspan_chord_m,
        b_ref=case.span_m,
    )
    euler = [izraelevitz_euler_spacing(index, case) for index in range(3)]
    amplitudes = tuple(item[0] for item in euler)
    spacings = tuple(item[1] for item in euler)
    wing_movements = [
        ps.movements.wing_movement.WingMovement(
            base_wing=wing,
            wing_cross_section_movements=section_movements,
            ampAngles_Gs_to_Wn_ixyz=amplitudes,
            periodAngles_Gs_to_Wn_ixyz=(case.period_s,) * 3,
            spacingAngles_Gs_to_Wn_ixyz=spacings,
            phaseAngles_Gs_to_Wn_ixyz=(90.0, 90.0, 90.0),
        )
        for wing, section_movements in zip(wings, all_section_movements)
    ]
    airplane_movement = ps.movements.airplane_movement.AirplaneMovement(
        base_airplane=airplane, wing_movements=wing_movements
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
        "grid_chord_semispan": [nc, ns],
        "steps_per_cycle": steps_per_cycle,
        "cycles": cycles,
        "max_wake_cycles": wake_cycles,
        "tip_chord_regularization_m": tip_chord,
        "tip_chord_regularization_fraction_cmid": tip_chord / case.midspan_chord_m,
        "stroke_euler_amplitudes_deg": list(amplitudes),
        "stroke_rotation": (
            "R_y(90-beta) R_x(Upsilon) R_y(-(90-beta)); converted exactly "
            "to Ptera intrinsic-XYZ Euler histories"
        ),
        "left_mirror_plane_offset_m": 1.0e-4,
    }
    return movement, metadata


def build_izraelevitz_fig11_movement(
    quality: str = "full",
    case: IzraelevitzHeavePitchCase = IZRAELEVITZ_2017_FIG11,
    settings: tuple[int, int, int, int] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Build the paper's unambiguous AR=3 heave-pitch Figure 11 case."""

    if settings is None:
        nc, ns, steps_per_cycle, cycles = (
            (2, 7, 24, 2) if quality == "smoke" else (4, 12, 24, 4)
        )
    else:
        nc, ns, steps_per_cycle, cycles = settings
    airfoil = ps.geometry.airfoil.Airfoil(name="naca0012")
    eta = np.sin(np.linspace(0.0, 0.5 * np.pi, ns + 1))
    span_nodes = case.semispan_m * eta
    chord_nodes = case.chord_m(span_nodes)
    tip_chord = 0.002 * case.midspan_chord_m
    chord_nodes[-1] = tip_chord
    sections = []
    for index, (span, chord) in enumerate(zip(span_nodes, chord_nodes)):
        if index == 0:
            offset = (0.0, 0.0, 0.0)
        else:
            # Keep x_LE+0.25c=0 along the complete span.
            delta_span = span - span_nodes[index - 1]
            delta_x_le = -0.25 * (chord - chord_nodes[index - 1])
            offset = (float(delta_x_le), float(delta_span), 0.0)
        sections.append(
            ps.geometry.wing_cross_section.WingCrossSection(
                airfoil=airfoil,
                chord=float(chord),
                Lp_Wcsp_Lpp=offset,
                num_spanwise_panels=1 if index < ns else None,
                spanwise_spacing="uniform" if index < ns else None,
                control_surface_symmetry_type="symmetric",
            )
        )
    wing = ps.geometry.wing.Wing(
        name="Izraelevitz 2017 Figure 11 AR3 elliptic wing",
        wing_cross_sections=sections,
        Ler_Gs_Cgs=(-0.25 * case.midspan_chord_m, 0.0, 0.0),
        symmetric=True,
        symmetryNormal_G=(0.0, 1.0, 0.0),
        symmetryPoint_G_Cg=(0.0, 0.0, 0.0),
        num_chordwise_panels=nc,
        chordwise_spacing="uniform",
    )
    airplane = ps.geometry.airplane.Airplane(
        wings=[wing],
        name="Izraelevitz 2017 Figure 11 heave-pitch",
        s_ref=case.area_m2,
        c_ref=case.midspan_chord_m,
        b_ref=case.span_m,
    )
    section_movements = [
        ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
            base_wing_cross_section=section
        )
        for section in sections
    ]
    wing_movement = ps.movements.wing_movement.WingMovement(
        base_wing=wing,
        wing_cross_section_movements=section_movements,
        ampLer_Gs_Cgs=(0.0, 0.0, case.heave_amplitude_m),
        periodLer_Gs_Cgs=(0.0, 0.0, case.period_s),
        phaseLer_Gs_Cgs=(0.0, 0.0, 90.0),
        ampAngles_Gs_to_Wn_ixyz=(0.0, abs(case.pitch_amplitude_deg), 0.0),
        periodAngles_Gs_to_Wn_ixyz=(0.0, case.period_s, 0.0),
        phaseAngles_Gs_to_Wn_ixyz=(0.0, 180.0, 0.0),
        rotationPointOffset_Gs_Ler=(0.25 * case.midspan_chord_m, 0.0, 0.0),
    )
    airplane_movement = ps.movements.airplane_movement.AirplaneMovement(
        base_airplane=airplane, wing_movements=[wing_movement]
    )
    operating_point = ps.operating_point.OperatingPoint(
        rho=case.rho_kg_m3,
        vCg__E=case.freestream_m_s,
        alpha=0.0,
        beta=0.0,
        nu=case.nu_m2_s,
    )
    requested_delta_time = (
        None if steps_per_cycle == 0 else case.period_s / steps_per_cycle
    )
    movement = ps.movements.movement.Movement(
        airplane_movements=[airplane_movement],
        operating_point_movement=ps.movements.operating_point_movement.OperatingPointMovement(
            base_operating_point=operating_point
        ),
        delta_time=requested_delta_time,
        num_cycles=cycles,
        max_wake_cycles=min(2, cycles),
    )
    return movement, {
        "grid_chord_semispan": [nc, ns],
        "steps_per_cycle": case.period_s / movement.delta_time,
        "requested_steps_per_cycle": steps_per_cycle,
        "cycles": cycles,
        "tip_chord_regularization_m": tip_chord,
        "quarter_chord_line_x_m": 0.0,
        "heave_law": "z=h*cos(omega*t)",
        "pitch_law": "theta=theta_max*sin(omega*t), about straight quarter chord",
    }


def build_izraelevitz_scherer_movement(
    theta_max_deg: float,
    phase_offset_deg: float,
    quality: str = "full",
    case: IzraelevitzSchererCase = IZRAELEVITZ_2017_FIG14_SCHERER,
    settings: tuple[int, int, int, int] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Build the Figure-14 rectangular Scherer heave-pitch experiment.

    Scherer's original report supplies the 4-inch chord, 12-inch span and
    symmetric NACA 63A015 section.  Ptera's zero-camber NACA0012 object is
    used only as the identical flat *mean-surface* adapter because UVLM does
    not represent thickness.
    """

    if theta_max_deg <= 0.0:
        raise ValueError("theta_max_deg must be positive")
    if settings is None:
        nc, ns, steps_per_cycle, cycles = (
            (2, 6, 64, 3) if quality == "smoke" else (4, 12, 128, 4)
        )
    else:
        nc, ns, steps_per_cycle, cycles = settings

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
    pivot_offset = case.pivot_fraction_chord * case.chord_m
    wing = ps.geometry.wing.Wing(
        name="Izraelevitz 2017 Figure 14 Scherer AR3 rectangular wing",
        wing_cross_sections=[root, tip],
        Ler_Gs_Cgs=(-pivot_offset, 0.0, 0.0),
        symmetric=True,
        symmetryNormal_G=(0.0, 1.0, 0.0),
        symmetryPoint_G_Cg=(0.0, 0.0, 0.0),
        num_chordwise_panels=nc,
        chordwise_spacing="uniform",
    )
    airplane = ps.geometry.airplane.Airplane(
        wings=[wing],
        name=(
            "Izraelevitz 2017 Figure 14 Scherer "
            f"theta={theta_max_deg:g}, psi={phase_offset_deg:g}"
        ),
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
    ptera_pitch_phase_deg = (float(phase_offset_deg) + 90.0 + 180.0) % 360.0 - 180.0
    if np.isclose(ptera_pitch_phase_deg, -180.0):
        ptera_pitch_phase_deg = 180.0
    wing_movement = ps.movements.wing_movement.WingMovement(
        base_wing=wing,
        wing_cross_section_movements=section_movements,
        ampLer_Gs_Cgs=(0.0, 0.0, case.heave_amplitude_m),
        periodLer_Gs_Cgs=(0.0, 0.0, case.period_s),
        phaseLer_Gs_Cgs=(0.0, 0.0, 90.0),
        ampAngles_Gs_to_Wn_ixyz=(0.0, float(theta_max_deg), 0.0),
        periodAngles_Gs_to_Wn_ixyz=(0.0, case.period_s, 0.0),
        # Ptera uses sine spacing.  Adding 90 deg implements paper Eq. (61):
        # theta=theta_max*cos(omega*t+psi).
        phaseAngles_Gs_to_Wn_ixyz=(0.0, ptera_pitch_phase_deg, 0.0),
        rotationPointOffset_Gs_Ler=(pivot_offset, 0.0, 0.0),
    )
    airplane_movement = ps.movements.airplane_movement.AirplaneMovement(
        base_airplane=airplane, wing_movements=[wing_movement]
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
    return movement, {
        "case_id": case.case_id,
        "grid_chord_semispan": [nc, ns],
        "steps_per_cycle": case.period_s / movement.delta_time,
        "requested_steps_per_cycle": steps_per_cycle,
        "cycles": cycles,
        "max_wake_cycles": min(2, cycles),
        "theta_max_deg": float(theta_max_deg),
        "phase_offset_deg": float(phase_offset_deg),
        "ptera_pitch_phase_deg": ptera_pitch_phase_deg,
        "pivot_fraction_chord": case.pivot_fraction_chord,
        "heave_law": "z=h*cos(omega*t)",
        "pitch_law": "theta=theta_max*cos(omega*t+psi)",
        "profile_drag_coefficient": case.profile_drag_coefficient,
        "airfoil_adapter": (
            "zero-camber Ptera mean surface for Scherer NACA 63A015; UVLM "
            "does not represent symmetric-section thickness"
        ),
        "tip_shape_adapter": "rectangular tip; Scherer reports slightly rounded tips",
        "profile_drag_source_conflict": (
            "main Cd0=0.057 follows Izraelevitz Figure 14; fixed sensitivity "
            "Cd0=0.027 follows Scherer 1968 static tests"
        ),
    }


def _extract_last_cycle(
    solver: Any,
    movement: Any,
    period_s: float,
    rho: float,
    speed: float,
    area: float,
    output_samples: int = 128,
) -> dict[str, np.ndarray | float]:
    step_ids, time_s, lift_n, thrust_n = [], [], [], []
    for step in range(solver.unsteady_problem.first_results_step, solver.num_steps):
        force = solver.steady_problems[step].airplanes[0].forces_W
        if force is None:
            continue
        step_ids.append(step)
        time_s.append(step * movement.delta_time)
        # Ptera wind axes: body force +x is thrust; +z is down.
        lift_n.append(-float(force[2]))
        thrust_n.append(float(force[0]))
    step_ids = np.asarray(step_ids, dtype=int)
    time_s = np.asarray(time_s)
    lift_n = np.asarray(lift_n)
    thrust_n = np.asarray(thrust_n)
    if time_s.size < 4 or not np.all(np.isfinite(lift_n + thrust_n)):
        raise FloatingPointError("solver returned an incomplete or non-finite history")
    steps_per_cycle = int(round(period_s / movement.delta_time))
    if steps_per_cycle < 2:
        raise FloatingPointError("fewer than two time steps per cycle")
    last_step = int(step_ids[-1])
    cycle_at_last = last_step * movement.delta_time / period_s
    if np.isclose(cycle_at_last, round(cycle_at_last), atol=1.0e-10):
        # Ptera normally includes the final repeated cycle boundary.  Exclude
        # it so a 3-cycle/100-step run uses steps 200..299, not 200..300 with
        # phase zero represented twice.
        cycle_end_step = last_step
    else:
        cycle_end_step = last_step + 1
    cycle_start_step = cycle_end_step - steps_per_cycle
    mask = (step_ids >= cycle_start_step) & (step_ids < cycle_end_step)
    if int(np.count_nonzero(mask)) != steps_per_cycle:
        raise FloatingPointError(
            "solver did not return one coherent non-duplicated final cycle"
        )
    phase = np.mod(time_s[mask] / period_s, 1.0)
    order = np.argsort(phase)
    target = np.arange(output_samples, dtype=float) / output_samples
    lift = np.interp(target, phase[order], lift_n[mask][order], period=1.0)
    thrust = np.interp(target, phase[order], thrust_n[mask][order], period=1.0)
    q_area = 0.5 * rho * speed**2 * area
    source_lift = lift_n[mask]
    source_thrust = thrust_n[mask]
    return {
        "phase": target,
        "lift_n": lift,
        "thrust_n": thrust,
        "CL": lift / q_area,
        "CT": thrust / q_area,
        "mean_lift_n": float(np.mean(source_lift)),
        "mean_thrust_n": float(np.mean(source_thrust)),
        "mean_CL": float(np.mean(source_lift / q_area)),
        "mean_CT": float(np.mean(source_thrust / q_area)),
        "source_cycle_sample_count": steps_per_cycle,
        "source_cycle_step_range": [cycle_start_step, cycle_end_step - 1],
    }


def run_model(
    movement: Any,
    model: str,
    *,
    period_s: float,
    rho: float,
    speed: float,
    area: float,
    output_samples: int = 128,
) -> dict[str, Any]:
    if model not in MODEL_SEMANTICS:
        raise ValueError(f"unknown model: {model}")
    problem = ps.problems.UnsteadyProblem(movement=movement, only_final_results=False)
    start = time.perf_counter()
    if model == "fluxv_uvpm":
        solver = UVPMHybridSolver(
            problem,
            max_particles=20000,
            stretch=False,
            free_wake=False,
        )
        solver.run(
            prescribed_wake=True,
            calculate_streamlines=False,
            show_progress=False,
        )
        particle_count = int(solver._vpm_field.np)
    else:
        solver = ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver(
            problem
        )
        solver.run(
            prescribed_wake=model == "ptera_prescribed_wake_uvlm",
            calculate_streamlines=False,
            show_progress=False,
        )
        particle_count = 0
    history = _extract_last_cycle(
        solver,
        movement,
        period_s,
        rho,
        speed,
        area,
        output_samples,
    )
    history["runtime_s"] = time.perf_counter() - start
    history["particle_count"] = particle_count
    history["model_semantics"] = MODEL_SEMANTICS[model]
    return history
