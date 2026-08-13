"""Source-audited Meng et al. (2025) rigid flapping--pitching case.

The paper calls the second rigid-body degree of freedom ``twist``.  Its own
mechanism definition and Figure 10 show that this is a uniform rotation of the
rigid wing about the main spar, rather than a published spanwise twist law.
The word ``pitch`` is therefore used for the aerodynamic implementation while
the paper's terminology is retained in manifests and public arguments.

Only dimensions actually printed in Figure 11 are treated as observations.
The rounded outer planform and the main-spar position are drawing-derived
adapters, and their status is recorded explicitly.  No NACA 0012/0013 section
is used: Ptera receives a custom zero-camber, numerically thin membrane
outline because Meng et al. do not publish an airfoil or thickness.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pterasoftware as ps


STANDARD_GRAVITY_M_S2 = 9.80665


@dataclass(frozen=True)
class Meng2025Case:
    """Nominal Figure-16 wind-tunnel condition and Figure-11 geometry."""

    case_id: str = "meng2025_fig16_rigid_flap_pitch"
    half_span_m: float = 0.800
    root_chord_m: float = 0.287
    constant_chord_span_m: float = 0.340
    main_spar_fraction_root_chord: float = 0.27
    flap_amplitude_peak_to_peak_deg: float = 45.0
    freestream_m_s: float = 8.0
    installation_aoa_deg: float = 5.0
    frequency_hz: float = 2.0
    rho_kg_m3: float = 1.225
    nu_m2_s: float = 1.50e-5

    @property
    def span_m(self) -> float:
        return 2.0 * self.half_span_m

    @property
    def outer_span_m(self) -> float:
        return self.half_span_m - self.constant_chord_span_m

    @property
    def area_m2(self) -> float:
        # Figure 11 shows a constant-chord 340 mm inner panel followed by a
        # rounded outer panel.  The latter is digitized as a quarter ellipse.
        half_area = self.root_chord_m * (
            self.constant_chord_span_m + 0.25 * np.pi * self.outer_span_m
        )
        return float(2.0 * half_area)

    @property
    def aspect_ratio(self) -> float:
        return self.span_m**2 / self.area_m2

    @property
    def mean_chord_m(self) -> float:
        return self.area_m2 / self.span_m

    @property
    def main_spar_offset_m(self) -> float:
        return self.main_spar_fraction_root_chord * self.root_chord_m

    @property
    def period_s(self) -> float:
        return 1.0 / self.frequency_hz

    @property
    def reynolds_root(self) -> float:
        return self.freestream_m_s * self.root_chord_m / self.nu_m2_s

    @property
    def reynolds_mean(self) -> float:
        return self.freestream_m_s * self.mean_chord_m / self.nu_m2_s

    def chord_m(self, span_coordinate_m: np.ndarray | float) -> np.ndarray:
        """Return the Figure-11 planform adapter at a semispan coordinate."""

        y = np.asarray(span_coordinate_m, dtype=float)
        if np.any(y < 0.0) or np.any(y > self.half_span_m):
            raise ValueError("span coordinate must lie on the published semispan")
        outer_eta = np.clip(
            (y - self.constant_chord_span_m) / self.outer_span_m, 0.0, 1.0
        )
        rounded = self.root_chord_m * np.sqrt(np.clip(1.0 - outer_eta**2, 0.0, None))
        return np.where(y <= self.constant_chord_span_m, self.root_chord_m, rounded)

    def manifest(self) -> dict[str, float | str]:
        out: dict[str, float | str] = asdict(self)
        out.update(
            span_m=self.span_m,
            area_m2=self.area_m2,
            aspect_ratio=self.aspect_ratio,
            mean_chord_m=self.mean_chord_m,
            main_spar_offset_m=self.main_spar_offset_m,
            period_s=self.period_s,
            reynolds_root=self.reynolds_root,
            reynolds_mean=self.reynolds_mean,
            geometry_observation=(
                "Figure 11(b): semispan 800 mm, root chord 287 mm, inner "
                "dimension 340 mm and a straight leading edge"
            ),
            planform_adapter=(
                "constant chord to y=340 mm plus quarter-ellipse outer "
                "trailing edge digitized from Figure 11(b)"
            ),
            main_spar_adapter=(
                "x/c_root=0.27 digitized from Figure 11(b); no numerical "
                "spar-axis coordinate is printed"
            ),
            density_assumption=(
                "rho=1.225 kg/m^3 standard-air assumption; paper gives no "
                "test-section density, temperature, or pressure"
            ),
            section_adapter=(
                "custom zero-camber numerically thin membrane mean surface; "
                "paper publishes no section or thickness"
            ),
        )
        return out


MENG_2025 = Meng2025Case()


def nominal_kinematics_deg(
    phase: np.ndarray | float,
    twist_amplitude_peak_to_peak_deg: float,
    case: Meng2025Case = MENG_2025,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(flap, pitch)`` for the source-defined nominal phase.

    ``phase=0`` is the upper stroke limit and ``0 < phase < 0.5`` is the
    downstroke, matching Figures 10 and 16.  Positive aerodynamic pitch means
    nose-up, so the paper's downstroke nose-down twist has a negative sign.
    Both published amplitudes are peak-to-peak values.
    """

    if not 0.0 <= twist_amplitude_peak_to_peak_deg <= 45.0:
        raise ValueError("Meng twist amplitude must lie in the tested 0..45 deg range")
    tau = 2.0 * np.pi * np.asarray(phase, dtype=float)
    flap = 0.5 * case.flap_amplitude_peak_to_peak_deg * np.cos(tau)
    pitch = -0.5 * twist_amplitude_peak_to_peak_deg * np.sin(tau)
    return flap, pitch


def balance_to_wind_axis(
    force_x: np.ndarray | float,
    force_z: np.ndarray | float,
    *,
    alpha_deg: float,
    gravity_force: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Execute Meng et al. Equation (11) with its printed parentheses.

    The gravity vector is added in load-cell coordinates *before* applying
    the rotation.  Consequently gravity contributes exactly ``+G`` to lift
    and exactly zero to net thrust for every angle of attack.
    """

    fx = np.asarray(force_x, dtype=float)
    fz = np.asarray(force_z, dtype=float)
    if fx.shape != fz.shape:
        raise ValueError("force_x and force_z must have the same shape")
    if gravity_force < 0.0 or not np.isfinite(gravity_force):
        raise ValueError("gravity_force must be finite and nonnegative")
    alpha = np.deg2rad(float(alpha_deg))
    sa, ca = np.sin(alpha), np.cos(alpha)
    corrected_x = fx + gravity_force * sa
    corrected_z = fz + gravity_force * ca
    lift = sa * corrected_x + ca * corrected_z
    net_thrust = ca * corrected_x - sa * corrected_z
    return lift, net_thrust


def _thin_membrane_airfoil() -> Any:
    # Ptera rejects a truly zero-thickness outline.  This 0.02%-thick symmetric
    # outline has an exactly zero mean camber line and is only a topology
    # carrier for the UVLM surface, not a NACA-thickness model.
    eps = 1.0e-4
    outline = np.asarray(
        [
            [1.0, eps],
            [0.5, eps],
            [0.0, 0.0],
            [0.5, -eps],
            [1.0, -eps],
        ]
    )
    return ps.geometry.airfoil.Airfoil(
        name="Meng-2025-zero-camber-thin-membrane-adapter",
        outline_A_lp=outline,
        resample=False,
    )


def build_meng2025_movement(
    twist_amplitude_peak_to_peak_deg: float,
    *,
    quality: str = "full",
    case: Meng2025Case = MENG_2025,
    settings: tuple[int, int, int, int, int] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Build the nominal rigid two-wing Meng Figure-16 movement."""

    if not 0.0 <= twist_amplitude_peak_to_peak_deg <= 45.0:
        raise ValueError("Meng twist amplitude must lie in the tested 0..45 deg range")
    if settings is None:
        if quality == "smoke":
            nc, ns, steps_per_cycle, cycles, wake_cycles = (2, 8, 24, 2, 1)
        elif quality == "full":
            nc, ns, steps_per_cycle, cycles, wake_cycles = (4, 16, 96, 4, 2)
        else:
            raise ValueError("quality must be 'smoke' or 'full'")
    else:
        nc, ns, steps_per_cycle, cycles, wake_cycles = settings
    if min(nc, ns, steps_per_cycle, cycles, wake_cycles) <= 0:
        raise ValueError("all discretization settings must be positive")
    if wake_cycles > cycles:
        raise ValueError("max_wake_cycles cannot exceed num_cycles")

    span_nodes = np.linspace(0.0, case.half_span_m, ns + 1)
    chord_nodes = case.chord_m(span_nodes)
    tip_chord = 0.002 * case.root_chord_m
    chord_nodes[-1] = tip_chord
    airfoil = _thin_membrane_airfoil()
    pivot = case.main_spar_offset_m
    wings = []
    all_sections = []
    for side_name, mirror_only in (("right", False), ("left", True)):
        sections = []
        for index, (span, chord) in enumerate(zip(span_nodes, chord_nodes)):
            offset_y = 0.0 if index == 0 else span - span_nodes[index - 1]
            sections.append(
                ps.geometry.wing_cross_section.WingCrossSection(
                    airfoil=airfoil,
                    chord=float(chord),
                    Lp_Wcsp_Lpp=(0.0, float(offset_y), 0.0),
                    num_spanwise_panels=1 if index < ns else None,
                    spanwise_spacing="uniform" if index < ns else None,
                )
            )
        wings.append(
            ps.geometry.wing.Wing(
                name=f"Meng 2025 Figure 11 rigid {side_name} half-wing",
                wing_cross_sections=sections,
                Ler_Gs_Cgs=(-pivot, 0.0, 0.0),
                angles_Gs_to_Wn_ixyz=(0.0, case.installation_aoa_deg, 0.0),
                symmetric=False,
                mirror_only=mirror_only,
                symmetryNormal_G=(0.0, 1.0, 0.0) if mirror_only else None,
                # Avoid Ptera switching its internal mirror type exactly as a
                # flapping half-wing crosses the global x-z plane.  This is the
                # same recorded 0.1 mm numerical offset used by the existing
                # Izraelevitz moving-wing adapter.
                symmetryPoint_G_Cg=(0.0, 1.0e-4, 0.0) if mirror_only else None,
                num_chordwise_panels=nc,
                chordwise_spacing="uniform",
            )
        )
        all_sections.append(sections)
    airplane = ps.geometry.airplane.Airplane(
        wings=wings,
        name=f"Meng 2025 Figure 16 twist={twist_amplitude_peak_to_peak_deg:g}",
        s_ref=case.area_m2,
        c_ref=case.mean_chord_m,
        b_ref=case.span_m,
    )
    twist_half = 0.5 * float(twist_amplitude_peak_to_peak_deg)
    wing_movements = []
    for wing, sections in zip(wings, all_sections):
        section_movements = [
            ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
                base_wing_cross_section=section
            )
            for section in sections
        ]
        wing_movements.append(
            ps.movements.wing_movement.WingMovement(
                base_wing=wing,
                wing_cross_section_movements=section_movements,
                ampAngles_Gs_to_Wn_ixyz=(
                    0.5 * case.flap_amplitude_peak_to_peak_deg,
                    twist_half,
                    0.0,
                ),
                periodAngles_Gs_to_Wn_ixyz=(
                    case.period_s,
                    case.period_s if twist_half else 0.0,
                    0.0,
                ),
                # Ptera uses sine spacing: +90 deg gives cos for flap; +180 deg
                # gives -sin for the paper's downstroke nose-down pitch.
                phaseAngles_Gs_to_Wn_ixyz=(
                    90.0,
                    180.0 if twist_half else 0.0,
                    0.0,
                ),
                rotationPointOffset_Gs_Ler=(pivot, 0.0, 0.0),
            )
        )
    airplane_movement = ps.movements.airplane_movement.AirplaneMovement(
        base_airplane=airplane,
        wing_movements=wing_movements,
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
    metadata: dict[str, Any] = {
        "case_id": case.case_id,
        "grid_chord_semispan": [nc, ns],
        "requested_steps_per_cycle": steps_per_cycle,
        "steps_per_cycle": case.period_s / movement.delta_time,
        "cycles": cycles,
        "max_wake_cycles": wake_cycles,
        "twist_amplitude_peak_to_peak_deg": float(twist_amplitude_peak_to_peak_deg),
        "pitch_half_amplitude_deg": twist_half,
        "flap_amplitude_peak_to_peak_deg": case.flap_amplitude_peak_to_peak_deg,
        "flap_half_amplitude_deg": 0.5 * case.flap_amplitude_peak_to_peak_deg,
        "phase_origin": "upper stroke limit; downstroke occupies 0<t/T<0.5",
        "flap_law": "phi=(45 deg/2)*cos(2*pi*t/T)",
        "pitch_law": "theta=-(twist_pp/2)*sin(2*pi*t/T), positive=nose-up",
        "published_motion_input_missing": (
            "per-condition encoder histories; nominal sinusoidal Figure-10 law used"
        ),
        "tip_chord_regularization_m": tip_chord,
        "left_mirror_plane_offset_m": 1.0e-4,
        "geometry_manifest": case.manifest(),
        "airfoil_adapter": airfoil.name,
    }
    return movement, metadata
