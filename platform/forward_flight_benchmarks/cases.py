"""Published geometry and kinematics for two rigid-wing forward-flight cases.

All dimensional choices not fixed by Izraelevitz et al. are an explicit
similarity scaling.  Yang et al.'s validation-wing dimensions and mechanism
are dimensional and are copied directly from the paper.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Callable

import numpy as np
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class IzraelevitzCase:
    case_id: str = "izraelevitz2017_fig13"
    aspect_ratio: float = 6.0
    midspan_chord_m: float = 0.100
    freestream_m_s: float = 1.0
    rho_kg_m3: float = 1.225
    nu_m2_s: float = 1.506e-5
    strouhal: float = 0.3
    flap_amplitude_deg: float = 30.0
    tip_twist_amplitude_deg: float = -30.0
    stroke_angle_beta_deg: float = 75.0
    paper_alpha_min_deg: float = -8.5
    paper_alpha_max_deg: float = 26.1

    @property
    def span_m(self) -> float:
        # For c(y)=c_mid sqrt(1-(2y/b)^2), S=pi*b*c_mid/4 and AR=b^2/S.
        return np.pi * self.aspect_ratio * self.midspan_chord_m / 4.0

    @property
    def semispan_m(self) -> float:
        return self.span_m / 2.0

    @property
    def area_m2(self) -> float:
        return np.pi * self.span_m * self.midspan_chord_m / 4.0

    @property
    def angular_frequency_rad_s(self) -> float:
        # Paper Eq. (59): St=(omega/(pi U))*s_tip*cos(Upsilon_max)*cos(beta).
        denominator = (
            self.semispan_m
            * np.cos(np.deg2rad(self.flap_amplitude_deg))
            * np.cos(np.deg2rad(self.stroke_angle_beta_deg))
        )
        return self.strouhal * np.pi * self.freestream_m_s / denominator

    @property
    def frequency_hz(self) -> float:
        return self.angular_frequency_rad_s / (2.0 * np.pi)

    @property
    def period_s(self) -> float:
        return 1.0 / self.frequency_hz

    @property
    def reduced_frequency_midspan(self) -> float:
        return (
            self.angular_frequency_rad_s
            * self.midspan_chord_m
            / (2.0 * self.freestream_m_s)
        )

    @property
    def stroke_axis_tilt_deg(self) -> float:
        # beta is the stroke-plane angle in Fig. 12; the axis is 90-beta from x.
        return 90.0 - self.stroke_angle_beta_deg

    def chord_m(self, span_coordinate_m: np.ndarray | float) -> np.ndarray:
        eta = np.asarray(span_coordinate_m, dtype=float) / self.semispan_m
        return self.midspan_chord_m * np.sqrt(np.clip(1.0 - eta**2, 0.0, None))

    def manifest(self) -> dict[str, float | str]:
        out = asdict(self)
        out.update(
            span_m=self.span_m,
            semispan_m=self.semispan_m,
            area_m2=self.area_m2,
            frequency_hz=self.frequency_hz,
            period_s=self.period_s,
            reduced_frequency_midspan=self.reduced_frequency_midspan,
            stroke_axis_tilt_deg=self.stroke_axis_tilt_deg,
            dimensional_scaling=(
                "c_mid=0.1 m and U=1 m/s are similarity choices; paper Fig. 13 "
                "is nondimensional and fixes AR, St and angular amplitudes."
            ),
        )
        return out


@dataclass(frozen=True)
class IzraelevitzHeavePitchCase:
    """Paper Figure 11: unambiguous AR=3 elliptic heave-pitch case."""

    case_id: str = "izraelevitz2017_fig11_heave_pitch"
    aspect_ratio: float = 3.0
    midspan_chord_m: float = 0.100
    freestream_m_s: float = 1.0
    rho_kg_m3: float = 1.225
    nu_m2_s: float = 1.506e-5
    strouhal: float = 0.3
    heave_to_chord: float = 0.75
    downstroke_midpoint_alpha_deg: float = 15.0

    @property
    def span_m(self) -> float:
        return np.pi * self.aspect_ratio * self.midspan_chord_m / 4.0

    @property
    def semispan_m(self) -> float:
        return self.span_m / 2.0

    @property
    def area_m2(self) -> float:
        return np.pi * self.span_m * self.midspan_chord_m / 4.0

    @property
    def heave_amplitude_m(self) -> float:
        return self.heave_to_chord * self.midspan_chord_m

    @property
    def angular_frequency_rad_s(self) -> float:
        # Paper Eq. (56): St=h*omega/(pi*U).
        return self.strouhal * np.pi * self.freestream_m_s / self.heave_amplitude_m

    @property
    def frequency_hz(self) -> float:
        return self.angular_frequency_rad_s / (2.0 * np.pi)

    @property
    def period_s(self) -> float:
        return 1.0 / self.frequency_hz

    @property
    def reduced_frequency_midspan(self) -> float:
        return (
            self.angular_frequency_rad_s
            * self.midspan_chord_m
            / (2.0 * self.freestream_m_s)
        )

    @property
    def pitch_amplitude_deg(self) -> float:
        # Paper Eq. (55), chosen to make alpha=15 deg at mid-downstroke.
        return float(
            -np.rad2deg(
                np.arctan(
                    self.angular_frequency_rad_s
                    * self.heave_amplitude_m
                    / self.freestream_m_s
                )
            )
            + self.downstroke_midpoint_alpha_deg
        )

    def chord_m(self, span_coordinate_m: np.ndarray | float) -> np.ndarray:
        eta = np.asarray(span_coordinate_m, dtype=float) / self.semispan_m
        return self.midspan_chord_m * np.sqrt(np.clip(1.0 - eta**2, 0.0, None))

    def manifest(self) -> dict[str, float | str]:
        out = asdict(self)
        out.update(
            span_m=self.span_m,
            semispan_m=self.semispan_m,
            area_m2=self.area_m2,
            heave_amplitude_m=self.heave_amplitude_m,
            frequency_hz=self.frequency_hz,
            period_s=self.period_s,
            reduced_frequency_midspan=self.reduced_frequency_midspan,
            pitch_amplitude_deg=self.pitch_amplitude_deg,
            dimensional_scaling=(
                "c_mid=0.1 m and U=1 m/s are similarity choices; the paper fixes "
                "AR=3, St=0.3, h/c_mid=0.75 and alpha_max=15 deg."
            ),
        )
        return out


@dataclass(frozen=True)
class IzraelevitzSchererCase:
    """Paper Figure 14: Scherer finite-wing experimental comparison.

    The dimensional geometry comes from Scherer's original 1968 report.  The
    operating speed below realizes its nondimensional ``J'=U/(f*c)=6`` at the
    rig's documented 5 Hz upper frequency; comparisons remain nondimensional.
    """

    case_id: str = "izraelevitz2017_fig14_scherer_experiment"
    aspect_ratio: float = 3.0
    chord_m: float = 0.1016
    freestream_m_s: float = 3.048
    rho_kg_m3: float = 1000.0
    nu_m2_s: float = 1.0e-6
    strouhal: float = 0.2
    heave_to_chord: float = 0.6
    profile_drag_coefficient: float = 0.057
    scherer_static_profile_drag_coefficient: float = 0.027
    pivot_fraction_chord: float = 0.75
    section_name: str = "NACA 63A015"

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
    def heave_amplitude_m(self) -> float:
        return self.heave_to_chord * self.chord_m

    @property
    def angular_frequency_rad_s(self) -> float:
        # Same Strouhal convention as paper Eq. (56): St=h*omega/(pi*U).
        return self.strouhal * np.pi * self.freestream_m_s / self.heave_amplitude_m

    @property
    def frequency_hz(self) -> float:
        return self.angular_frequency_rad_s / (2.0 * np.pi)

    @property
    def period_s(self) -> float:
        return 1.0 / self.frequency_hz

    @property
    def reduced_frequency_midspan(self) -> float:
        return self.angular_frequency_rad_s * self.chord_m / (2.0 * self.freestream_m_s)

    def manifest(self) -> dict[str, float | str]:
        out = asdict(self)
        out.update(
            span_m=self.span_m,
            semispan_m=self.semispan_m,
            area_m2=self.area_m2,
            heave_amplitude_m=self.heave_amplitude_m,
            angular_frequency_rad_s=self.angular_frequency_rad_s,
            frequency_hz=self.frequency_hz,
            period_s=self.period_s,
            reduced_frequency_midspan=self.reduced_frequency_midspan,
            dimensional_scaling=(
                "Scherer geometry: c=4 in, b=12 in, NACA 63A015, AR=3 and "
                "pivot=0.75c. U=10 ft/s realizes J'=6 at f=5 Hz; Figure 14 "
                "comparisons use nondimensional CT."
            ),
            profile_drag_source_conflict=(
                "Izraelevitz 2017 applies Cd0=0.057, whereas Scherer 1968 "
                "reports static CD0=0.027. The main reproduction follows "
                "Izraelevitz and a fixed 0.027 sensitivity is reported."
            ),
            geometry_adapter=(
                "Scherer describes slightly rounded rectangular tips. The UVLM "
                "uses a rectangular planform; symmetric NACA 63A015 has a flat "
                "mean camber surface, represented by Ptera's zero-camber adapter."
            ),
        )
        return out


@dataclass(frozen=True)
class YangCase:
    case_id: str = "yang2023_fig7_rigid"
    chord_m: float = 0.130
    span_m: float = 0.250
    thickness_m: float = 0.001
    freestream_m_s: float = 5.5
    frequency_hz: float = 2.5
    rho_kg_m3: float = 1.225
    nu_m2_s: float = 1.506e-5
    phi0_deg: float = 12.0
    fixed_link_m: float = 0.0376
    crank_m: float = 0.0100
    coupler_m: float = 0.0335
    rocker_m: float = 0.0200
    target_downstroke_deg: float = -15.0
    target_upstroke_deg: float = 45.0

    @property
    def area_m2(self) -> float:
        return self.chord_m * self.span_m

    @property
    def aspect_ratio(self) -> float:
        return self.span_m**2 / self.area_m2

    @property
    def period_s(self) -> float:
        return 1.0 / self.frequency_hz

    @property
    def reynolds(self) -> float:
        return self.freestream_m_s * self.chord_m / self.nu_m2_s

    def manifest(self) -> dict[str, float | str]:
        out = asdict(self)
        extrema = fourbar_extrema_deg(self)
        out.update(
            area_m2=self.area_m2,
            aspect_ratio=self.aspect_ratio,
            period_s=self.period_s,
            reynolds=self.reynolds,
            reconstructed_flap_min_deg=extrema[0],
            reconstructed_flap_max_deg=extrema[1],
        )
        return out


@dataclass(frozen=True)
class Yang2025RigidCase:
    """Peer-reviewed JFS 2025 rigid-wing validation configuration.

    The formal paper changed the validation mechanism from the IMAV precursor.
    Its reported ``phi0=-14.5 deg`` uses the opposite horizontal sign from the
    Fig. 9 coordinate convention used by :func:`fourbar_flap_angle_deg`; the
    converted value is exposed explicitly instead of silently changing the
    published parameter.
    """

    case_id: str = "yang2025_fig11_rigid"
    chord_m: float = 0.130
    span_m: float = 0.250
    thickness_m: float = 0.001
    wing_root_offset_m: float = 0.080
    freestream_m_s: float = 5.5
    frequency_hz: float = 2.5
    rho_kg_m3: float = 1.23
    nu_m2_s: float = 1.47e-5
    phi0_deg: float = -14.5
    fixed_link_m: float = 0.0479
    crank_m: float = 0.0084
    coupler_m: float = 0.0457
    rocker_m: float = 0.0144
    target_downstroke_deg: float = -30.0
    target_upstroke_deg: float = 40.0
    chordwise_panels: int = 8
    spanwise_panels: int = 12
    steps_per_cycle: int = 100
    separation_aoa_deg: float = 5.0
    aoa_range_coefficient: float = 5.0
    attached_plev_coefficient: float = 0.4
    separated_plev_coefficient: float = -0.8
    plev_core_radius_coefficient: float = 0.05
    squire_parameter: float = 0.001
    initial_core_radius_fraction_chord: float = 1.0e-5

    @property
    def mechanism_phi0_deg(self) -> float:
        return -self.phi0_deg

    @property
    def area_m2(self) -> float:
        return self.chord_m * self.span_m

    @property
    def aspect_ratio(self) -> float:
        return self.span_m**2 / self.area_m2

    @property
    def period_s(self) -> float:
        return 1.0 / self.frequency_hz

    @property
    def reynolds(self) -> float:
        return self.freestream_m_s * self.chord_m / self.nu_m2_s

    @property
    def initial_core_radius_m(self) -> float:
        return self.initial_core_radius_fraction_chord * self.chord_m

    def manifest(self) -> dict[str, float | str]:
        out = asdict(self)
        extrema = fourbar_extrema_deg(self)
        out.update(
            area_m2=self.area_m2,
            aspect_ratio=self.aspect_ratio,
            period_s=self.period_s,
            reynolds=self.reynolds,
            initial_core_radius_m=self.initial_core_radius_m,
            mechanism_phi0_deg=self.mechanism_phi0_deg,
            reconstructed_flap_min_deg=extrema[0],
            reconstructed_flap_max_deg=extrema[1],
            kinematics_provenance=(
                "Four-bar reconstruction from rounded JFS 2025 link lengths; "
                "the paper simulation used an unpublished laser-measured history."
            ),
        )
        return out


IZRAELEVITZ_2017 = IzraelevitzCase()
IZRAELEVITZ_2017_FIG11 = IzraelevitzHeavePitchCase()
IZRAELEVITZ_2017_FIG14_SCHERER = IzraelevitzSchererCase()
YANG_2023 = YangCase()
YANG_2025 = Yang2025RigidCase()


def _circle_intersection_left(
    centre_a: np.ndarray, radius_a: float, centre_b: np.ndarray, radius_b: float
) -> np.ndarray:
    """Return the open four-bar branch lying to the left of the wing joint."""

    delta = centre_b - centre_a
    distance = float(np.linalg.norm(delta))
    if not abs(radius_a - radius_b) <= distance <= radius_a + radius_b:
        raise ValueError("four-bar linkage cannot close")
    along = (radius_a**2 - radius_b**2 + distance**2) / (2.0 * distance)
    height = np.sqrt(max(radius_a**2 - along**2, 0.0))
    midpoint = centre_a + along * delta / distance
    perpendicular = np.array([-delta[1], delta[0]]) / distance
    candidates = (midpoint + height * perpendicular, midpoint - height * perpendicular)
    return min(candidates, key=lambda point: point[0])


def fourbar_flap_angle_deg(
    crank_phase_rad: np.ndarray | float,
    case: YangCase | Yang2025RigidCase = YANG_2023,
) -> np.ndarray:
    """Wing-extension angle produced by Yang's planar crank-rocker.

    Coordinates follow Fig. 5: horizontal is +Y, vertical is +Z, the lower
    ground pivot is the origin, and the upper wing joint is tilted by phi0
    from vertical.  The green wing is the extension of the orange rocker.
    """

    phase = np.asarray(crank_phase_rad, dtype=float)
    phi0_deg = getattr(case, "mechanism_phi0_deg", case.phi0_deg)
    fixed = np.array(
        [
            case.fixed_link_m * np.sin(np.deg2rad(phi0_deg)),
            case.fixed_link_m * np.cos(np.deg2rad(phi0_deg)),
        ]
    )
    result = np.empty_like(phase, dtype=float)
    for index in np.ndindex(phase.shape):
        q = float(phase[index])
        crank_end = case.crank_m * np.array([np.cos(q), np.sin(q)])
        coupler_rocker_joint = _circle_intersection_left(
            crank_end, case.coupler_m, fixed, case.rocker_m
        )
        wing_direction = (fixed - coupler_rocker_joint) / case.rocker_m
        result[index] = np.rad2deg(np.arctan2(wing_direction[1], wing_direction[0]))
    return result


@lru_cache(maxsize=4)
def fourbar_extrema_deg(
    case: YangCase | Yang2025RigidCase = YANG_2023,
) -> tuple[float, float]:
    phase = np.linspace(0.0, 2.0 * np.pi, 20001)
    angles = fourbar_flap_angle_deg(phase, case)
    return float(np.min(angles)), float(np.max(angles))


@lru_cache(maxsize=4)
def fourbar_zero_phase_rad(
    case: YangCase | Yang2025RigidCase = YANG_2023,
) -> float:
    lower, upper = fourbar_extrema_deg(case)
    mid = 0.5 * (lower + upper)
    # Choose the rising mean-angle crossing so a cycle starts at the base angle.
    phase = np.linspace(0.0, 2.0 * np.pi, 2001)
    residual = fourbar_flap_angle_deg(phase, case) - mid
    roots: list[float] = []
    for left, right, f_left, f_right in zip(
        phase[:-1], phase[1:], residual[:-1], residual[1:]
    ):
        if f_left * f_right < 0.0:
            roots.append(
                brentq(
                    lambda value: float(fourbar_flap_angle_deg(value, case) - mid),
                    left,
                    right,
                )
            )
    for root in roots:
        epsilon = 1.0e-5
        slope = float(
            fourbar_flap_angle_deg(root + epsilon, case)
            - fourbar_flap_angle_deg(root - epsilon, case)
        )
        if slope > 0.0:
            return root
    raise RuntimeError("could not locate rising four-bar mean crossing")


def fourbar_normalized_spacing(
    phase_rad: np.ndarray,
    case: YangCase | Yang2025RigidCase = YANG_2023,
) -> np.ndarray:
    """Unit-amplitude, 2-pi-periodic four-bar waveform."""

    lower, upper = fourbar_extrema_deg(case)
    mid = 0.5 * (lower + upper)
    amplitude = 0.5 * (upper - lower)
    phase = np.mod(np.asarray(phase_rad, dtype=float), 2.0 * np.pi)
    return (
        fourbar_flap_angle_deg(phase + fourbar_zero_phase_rad(case), case) - mid
    ) / amplitude


def yang_fourbar_spacing(phase_rad: np.ndarray) -> np.ndarray:
    """IMAV 2023 four-bar waveform retained for the original benchmark."""

    return fourbar_normalized_spacing(phase_rad, YANG_2023)


def yang2025_fourbar_spacing(phase_rad: np.ndarray) -> np.ndarray:
    """JFS 2025 four-bar reconstruction (not the unpublished LDS history)."""

    return fourbar_normalized_spacing(phase_rad, YANG_2025)


def _rotation_x(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rotation_y(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def izraelevitz_stroke_matrix(
    phase_rad: float, case: IzraelevitzCase = IZRAELEVITZ_2017
) -> np.ndarray:
    """Active rotation for a flap about the paper's tilted stroke axis.

    beta is drawn as the stroke-plane angle in Fig. 12.  The corresponding
    rotation-axis tilt from +x is delta=90-beta.  Conjugating Rx by Ry(delta)
    leaves the neutral wing/chord aligned with the freestream while tilting its
    flapping trajectory in x-z.
    """

    delta = np.deg2rad(case.stroke_axis_tilt_deg)
    upsilon = np.deg2rad(case.flap_amplitude_deg) * np.cos(phase_rad)
    return _rotation_y(delta) @ _rotation_x(upsilon) @ _rotation_y(-delta)


@lru_cache(maxsize=4)
def izraelevitz_euler_spacing(
    component: int, case: IzraelevitzCase = IZRAELEVITZ_2017
) -> tuple[float, Callable[[np.ndarray], np.ndarray]]:
    """Return amplitude and custom waveform for Ptera intrinsic-XYZ angles."""

    grid = np.linspace(0.0, 2.0 * np.pi, 4001)
    # Raw waveform uses sin(q), hence starts at the neutral wing. Ptera applies
    # a +90 degree phase to recover the paper's cos(omega*t) flap motion.
    delta = np.deg2rad(case.stroke_axis_tilt_deg)
    rows = []
    for q in grid:
        upsilon = np.deg2rad(case.flap_amplitude_deg) * np.sin(q)
        matrix = _rotation_y(delta) @ _rotation_x(upsilon) @ _rotation_y(-delta)
        rows.append(Rotation.from_matrix(matrix).as_euler("XYZ", degrees=True))
    values = np.asarray(rows)[:, component]
    amplitude = 0.5 * (float(np.max(values)) - float(np.min(values)))
    # Ptera requires every raw custom waveform to start at zero.  A vertical
    # shift does not change its peak-to-peak amplitude, so reference the
    # neutral-stroke Euler angle rather than the midrange.
    normalized = (values - values[0]) / amplitude

    def spacing(phase_rad: np.ndarray) -> np.ndarray:
        phase = np.mod(np.asarray(phase_rad, dtype=float), 2.0 * np.pi)
        return np.interp(phase, grid, normalized)

    return amplitude, spacing


def izraelevitz_tip_alpha_history(
    samples: int = 2048, case: IzraelevitzCase = IZRAELEVITZ_2017
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstructed local geometric alpha at the tip three-quarter chord.

    This diagnostic exposes the rotation-convention consequence; it is not used
    to alter or fit the aerodynamic solver input.
    """

    phase = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
    positions, rotations = [], []
    for q in phase:
        stroke = izraelevitz_stroke_matrix(q, case)
        twist = np.deg2rad(case.tip_twist_amplitude_deg) * np.sin(q)
        rotation = stroke @ _rotation_y(twist)
        rotations.append(rotation)
        positions.append(
            rotation @ np.array([0.75 * case.midspan_chord_m, case.semispan_m, 0.0])
        )
    positions = np.asarray(positions)
    time_s = phase / case.angular_frequency_rad_s
    velocity = np.gradient(positions, time_s, axis=0, edge_order=2)
    alpha = []
    for rotation, point_velocity in zip(rotations, velocity):
        chord_hat = rotation[:, 0]
        span_hat = rotation[:, 1]
        normal_hat = np.cross(chord_hat, span_hat)
        relative = np.array([case.freestream_m_s, 0.0, 0.0]) - point_velocity
        relative -= np.dot(relative, span_hat) * span_hat
        alpha.append(
            np.rad2deg(
                np.arctan2(np.dot(relative, normal_hat), np.dot(relative, chord_hat))
            )
        )
    return phase / (2.0 * np.pi), np.asarray(alpha)
