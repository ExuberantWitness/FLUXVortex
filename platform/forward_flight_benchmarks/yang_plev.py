"""Auditable Yang et al. (2025) modified-UVLM + PLEV core.

This module implements the equations and numerical parameters published in

    H.-H. Yang et al., *Journal of Fluids and Structures* 133 (2025) 104263,
    doi:10.1016/j.jfluidstructs.2024.104263.

The implementation is deliberately independent of PteraSoftware.  Yang's
bound vortex rings coincide with panel edges and the no-penetration condition
is imposed at panel centres, whereas Ptera's production UVLM uses a different
quarter-chord/three-quarter-chord layout.

Published equations implemented here
------------------------------------
* Eq. (8): viscous vortex-core growth.
* Eq. (9): Squire eddy-viscosity coefficient.
* Eq. (10): finite-core straight-segment Biot--Savart velocity.
* Eq. (11): ``Gamma_PLEV = K_PLEV Gamma_LE``.
* Eq. (12): the piecewise local-angle PLEV coefficient.
* Eq. (13)--(14): the optional adaptive-wake circulation split.

Explicit reconstruction conventions
-----------------------------------
The paper leaves a few implementation details unstated or typographically
ambiguous.  They are kept in :data:`IMPLEMENTATION_CONVENTIONS` and copied to
every step result:

1. Printed Eq. (10) is dimensionally scalar although ``V`` is a vector.  The
   standard ``r1 x r2`` direction factor is included.
2. Eq. (9) is evaluated with ``abs(Gamma)/nu``.  A signed circulation would
   permit a negative eddy viscosity for one vortex orientation.
3. Eq. (12) tests ``abs(alpha_LE)`` in every branch but prints ``alpha_LE``
   inside the sine.  The magnitude is used inside the sine as implied by the
   branch conditions, Fig. 3, and the statement that the first lobe is
   positive.  The sign remains in ``Gamma_LE`` in Eq. (11).
4. ``v_LE`` is the magnitude of the known local wing-relative velocity before
   bound/PLEV self-induction.  This makes Eq. (11) a linear, simultaneously
   coupled correction to the bound AIC system.
5. The PLEV trailing line follows the local surface chord direction.  A PLEV
   is an instantaneous attached pseudo ring; it is not added to the material
   free wake.

The module is a circulation/influence core.  It intentionally does not claim
the paper's full load path or free-wake/AWS time integration; those require
additional conventions beyond Eqs. (8)--(14).  Optional externally managed
wake rings can nevertheless feed back through Eq. (8)--(10).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


IMPLEMENTATION_CONVENTIONS: tuple[str, ...] = (
    "Eq.10 includes the standard (r1 x r2) vector direction factor omitted "
    "from the printed typesetting.",
    "Eq.9 uses Re_V=abs(Gamma)/nu so vortex-core growth is independent of "
    "circulation orientation.",
    "Eq.12 uses abs(alpha_LE) both in its branch tests and sine arguments; "
    "Gamma_LE retains the circulation sign in Eq.11.",
    "v_LE is ||V_known - V_surface|| before bound and PLEV self-induction; "
    "spanwise velocity contributes to d_PLEV but not to alpha_LE.",
    "The PLEV trailing line is offset along the local surface chord and the "
    "PLEV is rebuilt each step rather than convected as material wake.",
    "PLEV feedback is simultaneous and linear: each leading-edge AIC column "
    "is A_bound + K_PLEV*A_PLEV.",
    "Eq.14 is read as exp(-6.27*dot), as confirmed by its printed 0.9981 "
    "normalizer; dot products outside the stated 0--90 degree range are clipped.",
)

MODEL_SCOPE_LIMITATIONS: tuple[str, ...] = (
    "Material free-wake birth, convection, roll-up, and truncation are not "
    "advanced internally; optional wake rings are caller-managed inputs.",
    "Eqs.13--14 expose the AWS strength split but do not select shedding edges "
    "or create an AWS wake topology.",
    "The paper's Eqs.6--7 pressure/load integration is outside this circulation "
    "core; no aerodynamic-force accuracy claim is made here.",
    "solve_history carries previous bound circulation for later unsteady terms "
    "but is not by itself the paper's complete 100-step-per-cycle simulation.",
)


@dataclass(frozen=True)
class Yang2025Parameters:
    """Published aerodynamic parameters, primarily from Table 1."""

    density_kg_m3: float = 1.23
    kinematic_viscosity_m2_s: float = 1.47e-5
    lamb_constant: float = 1.25643
    squire_parameter: float = 0.001
    initial_core_radius_fraction_mean_chord: float = 1.0e-5
    steps_per_flapping_cycle: int = 100
    chordwise_panels: int = 8
    spanwise_panels: int = 12
    wake_distance_fraction_coefficient: float = 5.0
    separation_angle_deg: float = 5.0
    aoa_range_coefficient: float = 5.0
    attached_plev_coefficient: float = 0.4
    separated_plev_coefficient: float = -0.8
    plev_core_radius_coefficient: float = 0.05
    adaptive_wake_exponent: float = 6.27

    def __post_init__(self) -> None:
        if self.density_kg_m3 <= 0.0:
            raise ValueError("density_kg_m3 must be positive")
        if self.kinematic_viscosity_m2_s <= 0.0:
            raise ValueError("kinematic_viscosity_m2_s must be positive")
        if self.lamb_constant <= 0.0:
            raise ValueError("lamb_constant must be positive")
        if self.squire_parameter < 0.0:
            raise ValueError("squire_parameter must be non-negative")
        if self.initial_core_radius_fraction_mean_chord <= 0.0:
            raise ValueError("initial core-radius fraction must be positive")
        if self.steps_per_flapping_cycle < 1:
            raise ValueError("steps_per_flapping_cycle must be positive")
        if self.chordwise_panels < 1 or self.spanwise_panels < 1:
            raise ValueError("panel counts must be positive")
        if not 0.0 <= self.attached_plev_coefficient <= 1.0:
            raise ValueError("attached_plev_coefficient must lie in [0, 1]")
        if not -1.0 <= self.separated_plev_coefficient <= 0.0:
            raise ValueError("separated_plev_coefficient must lie in [-1, 0]")
        if self.plev_core_radius_coefficient <= 0.0:
            raise ValueError("plev_core_radius_coefficient must be positive")
        threshold = self.aoa_range_coefficient * self.separation_angle_rad
        if not self.separation_angle_rad < threshold < 0.5 * np.pi:
            raise ValueError("C_alpha*alpha_sep must lie between alpha_sep and pi/2")

    @property
    def separation_angle_rad(self) -> float:
        return float(np.deg2rad(self.separation_angle_deg))

    def initial_core_radius_m(self, mean_chord_m: float) -> float:
        if mean_chord_m <= 0.0:
            raise ValueError("mean_chord_m must be positive")
        return self.initial_core_radius_fraction_mean_chord * mean_chord_m


YANG_2025_PARAMETERS = Yang2025Parameters()


def _scalar_or_array(value: np.ndarray) -> float | np.ndarray:
    if np.asarray(value).ndim == 0:
        return float(np.asarray(value))
    return value


def eddy_viscosity_coefficient(
    circulation_m2_s: float | np.ndarray,
    kinematic_viscosity_m2_s: float,
    squire_parameter: float = YANG_2025_PARAMETERS.squire_parameter,
) -> float | np.ndarray:
    """Return Eq. (9), using the documented ``abs(Gamma)`` convention."""

    if kinematic_viscosity_m2_s <= 0.0:
        raise ValueError("kinematic_viscosity_m2_s must be positive")
    if squire_parameter < 0.0:
        raise ValueError("squire_parameter must be non-negative")
    circulation = np.asarray(circulation_m2_s, dtype=float)
    reynolds_vortex = np.abs(circulation) / kinematic_viscosity_m2_s
    result = 1.0 + squire_parameter * reynolds_vortex
    return _scalar_or_array(result)


def vortex_core_radius(
    initial_core_radius_m: float,
    wake_age_s: float | np.ndarray,
    circulation_m2_s: float | np.ndarray,
    kinematic_viscosity_m2_s: float,
    *,
    lamb_constant: float = YANG_2025_PARAMETERS.lamb_constant,
    squire_parameter: float = YANG_2025_PARAMETERS.squire_parameter,
) -> float | np.ndarray:
    """Return the vortex-core radius from Eqs. (8) and (9)."""

    if initial_core_radius_m < 0.0:
        raise ValueError("initial_core_radius_m must be non-negative")
    if lamb_constant <= 0.0:
        raise ValueError("lamb_constant must be positive")
    age = np.asarray(wake_age_s, dtype=float)
    if np.any(age < 0.0):
        raise ValueError("wake_age_s must be non-negative")
    delta = eddy_viscosity_coefficient(
        circulation_m2_s,
        kinematic_viscosity_m2_s,
        squire_parameter,
    )
    result = np.sqrt(
        initial_core_radius_m**2
        + 4.0 * lamb_constant * kinematic_viscosity_m2_s * np.asarray(delta) * age
    )
    return _scalar_or_array(result)


def vortex_segment_velocity(
    points_m: np.ndarray | Sequence[float] | Sequence[Sequence[float]],
    start_m: np.ndarray | Sequence[float],
    end_m: np.ndarray | Sequence[float],
    circulation_m2_s: float,
    core_radius_m: float,
) -> np.ndarray:
    """Finite-core straight-segment velocity from Eq. (10).

    ``r1`` and ``r2`` are the start and end positions relative to each target
    point and ``r0 = r1-r2``, exactly as defined below Eq. (10).  The standard
    vector factor ``r1 x r2`` is included because it is absent from the printed
    equation even though the left-hand side is a vector.
    """

    original = np.asarray(points_m, dtype=float)
    one_point = original.ndim == 1
    points = np.atleast_2d(original)
    start = np.asarray(start_m, dtype=float)
    end = np.asarray(end_m, dtype=float)
    if points.shape[1:] != (3,) or start.shape != (3,) or end.shape != (3,):
        raise ValueError(
            "points, start, and end must contain three-dimensional vectors"
        )
    if core_radius_m < 0.0:
        raise ValueError("core_radius_m must be non-negative")

    r1 = start[None, :] - points
    r2 = end[None, :] - points
    r0 = start - end
    segment_length = float(np.linalg.norm(r0))
    velocity = np.zeros_like(points)
    if segment_length <= np.finfo(float).eps or circulation_m2_s == 0.0:
        return velocity[0] if one_point else velocity

    cross = np.cross(r1, r2)
    cross_squared = np.einsum("ij,ij->i", cross, cross)
    denominator = cross_squared + (core_radius_m * segment_length) ** 2
    norm_r1 = np.linalg.norm(r1, axis=1)
    norm_r2 = np.linalg.norm(r2, axis=1)
    valid = (
        (norm_r1 > np.finfo(float).eps)
        & (norm_r2 > np.finfo(float).eps)
        & (denominator > np.finfo(float).tiny)
    )
    if np.any(valid):
        endpoint_factor = np.zeros(points.shape[0], dtype=float)
        endpoint_factor[valid] = (
            np.einsum("j,ij->i", r0, r1[valid]) / norm_r1[valid]
            - np.einsum("j,ij->i", r0, r2[valid]) / norm_r2[valid]
        )
        velocity[valid] = (
            circulation_m2_s
            / (4.0 * np.pi)
            * cross[valid]
            * (endpoint_factor[valid] / denominator[valid])[:, None]
        )
    return velocity[0] if one_point else velocity


def vortex_ring_velocity(
    points_m: np.ndarray | Sequence[float] | Sequence[Sequence[float]],
    corners_m: np.ndarray | Sequence[Sequence[float]],
    circulation_m2_s: float,
    core_radius_m: float,
) -> np.ndarray:
    """Velocity induced by four sequential Eq. (10) vortex segments."""

    corners = np.asarray(corners_m, dtype=float)
    if corners.shape != (4, 3):
        raise ValueError("corners_m must have shape (4, 3)")
    points = np.asarray(points_m, dtype=float)
    velocity = np.zeros_like(points, dtype=float)
    for index in range(4):
        velocity += vortex_segment_velocity(
            points,
            corners[index],
            corners[(index + 1) % 4],
            circulation_m2_s,
            core_radius_m,
        )
    return velocity


def plev_strength_coefficient(
    alpha_le_rad: float | np.ndarray,
    parameters: Yang2025Parameters = YANG_2025_PARAMETERS,
) -> float | np.ndarray:
    """Return the piecewise PLEV strength parameter from Eq. (12)."""

    alpha = np.asarray(alpha_le_rad, dtype=float)
    magnitude = np.abs(alpha)
    if np.any(magnitude > 0.5 * np.pi + 1.0e-12):
        raise ValueError("Eq.12 is published only for |alpha_LE| <= pi/2")

    alpha_sep = parameters.separation_angle_rad
    alpha_transition = parameters.aoa_range_coefficient * alpha_sep
    coefficient = np.zeros_like(magnitude)

    attached = (magnitude > alpha_sep) & (magnitude <= alpha_transition)
    coefficient[attached] = parameters.attached_plev_coefficient * np.sin(
        np.pi
        * (magnitude[attached] - alpha_sep)
        / ((parameters.aoa_range_coefficient - 1.0) * alpha_sep)
    )

    separated = (magnitude > alpha_transition) & (magnitude <= 0.5 * np.pi)
    coefficient[separated] = parameters.separated_plev_coefficient * np.sin(
        (0.5 * np.pi)
        * (magnitude[separated] - alpha_transition)
        / (0.5 * np.pi - alpha_transition)
    )
    return _scalar_or_array(coefficient)


def plev_circulation(
    alpha_le_rad: float | np.ndarray,
    leading_edge_circulation_m2_s: float | np.ndarray,
    parameters: Yang2025Parameters = YANG_2025_PARAMETERS,
) -> float | np.ndarray:
    """Return ``Gamma_PLEV`` from Eqs. (11) and (12)."""

    result = np.asarray(
        plev_strength_coefficient(alpha_le_rad, parameters)
    ) * np.asarray(
        leading_edge_circulation_m2_s,
        dtype=float,
    )
    if (
        np.asarray(alpha_le_rad).ndim == 0
        and np.asarray(leading_edge_circulation_m2_s).ndim == 0
    ):
        return float(result)
    return result


def adaptive_wake_coefficient(
    local_velocity: np.ndarray | Sequence[float],
    shedding_direction: np.ndarray | Sequence[float],
    *,
    exponent: float = YANG_2025_PARAMETERS.adaptive_wake_exponent,
) -> float:
    """Return the AWS coefficient from Eq. (14).

    The printed denominator ``0.9981`` disambiguates the superscript: Eq. (14)
    is evaluated as ``(1-exp(-exponent*x))/(1-exp(-exponent))``, where ``x`` is
    the dot product of the two unit vectors.
    """

    velocity = np.asarray(local_velocity, dtype=float)
    direction = np.asarray(shedding_direction, dtype=float)
    if velocity.shape != (3,) or direction.shape != (3,):
        raise ValueError("local_velocity and shedding_direction must have shape (3,)")
    if exponent <= 0.0:
        raise ValueError("exponent must be positive")
    velocity_norm = float(np.linalg.norm(velocity))
    direction_norm = float(np.linalg.norm(direction))
    if velocity_norm <= np.finfo(float).eps or direction_norm <= np.finfo(float).eps:
        raise ValueError("AWS vectors must be nonzero")
    cosine = float(np.dot(velocity / velocity_norm, direction / direction_norm))
    cosine = float(np.clip(cosine, 0.0, 1.0))
    return float((1.0 - np.exp(-exponent * cosine)) / (1.0 - np.exp(-exponent)))


def adaptive_wake_split(
    pre_shedding_circulation_m2_s: float,
    wake_coefficient: float,
) -> tuple[float, float]:
    """Return ``(Gamma_WSE, Gamma_W)`` satisfying Eqs. (13)--(14)."""

    if not 0.0 <= wake_coefficient <= 1.0:
        raise ValueError("wake_coefficient must lie in [0, 1]")
    wake = wake_coefficient * pre_shedding_circulation_m2_s
    retained = pre_shedding_circulation_m2_s - wake
    return float(retained), float(wake)


@dataclass(frozen=True)
class PanelEdgeGeometry:
    """Panel-edge rings and panel-centre collocation geometry."""

    corners_m: np.ndarray
    collocation_m: np.ndarray
    normals: np.ndarray
    areas_m2: np.ndarray
    chord_directions: np.ndarray
    chord_lengths_m: np.ndarray

    @property
    def chordwise_panels(self) -> int:
        return int(self.corners_m.shape[0])

    @property
    def spanwise_panels(self) -> int:
        return int(self.corners_m.shape[1])


def rectangular_wing_vertices(
    chord_m: float,
    span_m: float,
    *,
    chordwise_panels: int = YANG_2025_PARAMETERS.chordwise_panels,
    spanwise_panels: int = YANG_2025_PARAMETERS.spanwise_panels,
    origin_m: Sequence[float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Return a flat rectangular wing grid, LE-to-TE by root-to-tip."""

    if chord_m <= 0.0 or span_m <= 0.0:
        raise ValueError("chord_m and span_m must be positive")
    if chordwise_panels < 1 or spanwise_panels < 1:
        raise ValueError("panel counts must be positive")
    origin = np.asarray(origin_m, dtype=float)
    if origin.shape != (3,):
        raise ValueError("origin_m must have shape (3,)")
    x = np.linspace(0.0, chord_m, chordwise_panels + 1)
    y = np.linspace(0.0, span_m, spanwise_panels + 1)
    vertices = np.zeros((chordwise_panels + 1, spanwise_panels + 1, 3))
    vertices[..., 0] = x[:, None]
    vertices[..., 1] = y[None, :]
    vertices += origin
    return vertices


def panel_edge_geometry(vertices_m: np.ndarray) -> PanelEdgeGeometry:
    """Build Yang's panel-edge rings and centre collocation points."""

    vertices = np.asarray(vertices_m, dtype=float)
    if vertices.ndim != 3 or vertices.shape[2] != 3:
        raise ValueError("vertices_m must have shape (M+1, N+1, 3)")
    chordwise = vertices.shape[0] - 1
    spanwise = vertices.shape[1] - 1
    if chordwise < 1 or spanwise < 1:
        raise ValueError("vertices_m must describe at least one panel")

    corners = np.empty((chordwise, spanwise, 4, 3), dtype=float)
    collocation = np.empty((chordwise, spanwise, 3), dtype=float)
    normals = np.empty((chordwise, spanwise, 3), dtype=float)
    areas = np.empty((chordwise, spanwise), dtype=float)
    chord_directions = np.empty((chordwise, spanwise, 3), dtype=float)
    chord_lengths = np.empty((chordwise, spanwise), dtype=float)

    for i in range(chordwise):
        for j in range(spanwise):
            front_left = vertices[i, j]
            back_left = vertices[i + 1, j]
            back_right = vertices[i + 1, j + 1]
            front_right = vertices[i, j + 1]
            ring = np.asarray((front_left, back_left, back_right, front_right))
            corners[i, j] = ring
            collocation[i, j] = np.mean(ring, axis=0)

            chord_vector = 0.5 * ((back_left - front_left) + (back_right - front_right))
            span_vector = 0.5 * ((front_right - front_left) + (back_right - back_left))
            chord_length = float(np.linalg.norm(chord_vector))
            normal_vector = np.cross(chord_vector, span_vector)
            normal_length = float(np.linalg.norm(normal_vector))
            if (
                chord_length <= np.finfo(float).eps
                or normal_length <= np.finfo(float).eps
            ):
                raise ValueError("degenerate panel geometry")
            chord_directions[i, j] = chord_vector / chord_length
            chord_lengths[i, j] = chord_length
            normals[i, j] = normal_vector / normal_length
            areas[i, j] = 0.5 * (
                np.linalg.norm(
                    np.cross(back_left - front_left, front_right - front_left)
                )
                + np.linalg.norm(
                    np.cross(back_right - back_left, front_right - back_left)
                )
            )

    return PanelEdgeGeometry(
        corners_m=corners,
        collocation_m=collocation,
        normals=normals,
        areas_m2=areas,
        chord_directions=chord_directions,
        chord_lengths_m=chord_lengths,
    )


@dataclass(frozen=True)
class WakeRing:
    """Externally managed material wake ring used by Eq. (8)--(10)."""

    corners_m: np.ndarray
    circulation_m2_s: float
    age_s: float
    initial_core_radius_m: float | None = None


@dataclass(frozen=True)
class YangPLEVStepResult:
    """Complete auditable state returned by one coupled solve."""

    step_index: int
    delta_time_s: float
    geometry: PanelEdgeGeometry
    known_relative_velocity_m_s: np.ndarray
    alpha_le_rad: np.ndarray
    k_plev: np.ndarray
    d_plev_m: np.ndarray
    r_plev_m: np.ndarray
    plev_corners_m: np.ndarray
    bound_circulation_m2_s: np.ndarray
    plev_circulation_m2_s: np.ndarray
    previous_bound_circulation_m2_s: np.ndarray
    bound_aic: np.ndarray
    plev_unit_aic: np.ndarray
    coupled_aic: np.ndarray
    rhs_normal_velocity_m_s: np.ndarray
    max_abs_normal_residual_m_s: float
    coupled_aic_condition_number: float
    implementation_conventions: tuple[str, ...] = IMPLEMENTATION_CONVENTIONS
    model_scope_limitations: tuple[str, ...] = MODEL_SCOPE_LIMITATIONS

    @property
    def circulation_rate_m2_s2(self) -> np.ndarray:
        """Backward-difference bound-circulation rate for this step."""

        return (
            self.bound_circulation_m2_s - self.previous_bound_circulation_m2_s
        ) / self.delta_time_s


class YangPLEVSolver:
    """Panel-edge rectangular/general-grid UVLM with simultaneous PLEV feedback."""

    def __init__(
        self,
        *,
        delta_time_s: float,
        mean_chord_m: float,
        freestream_velocity_m_s: Sequence[float] = (1.0, 0.0, 0.0),
        parameters: Yang2025Parameters = YANG_2025_PARAMETERS,
    ) -> None:
        if delta_time_s <= 0.0:
            raise ValueError("delta_time_s must be positive")
        if mean_chord_m <= 0.0:
            raise ValueError("mean_chord_m must be positive")
        freestream = np.asarray(freestream_velocity_m_s, dtype=float)
        if freestream.shape != (3,):
            raise ValueError("freestream_velocity_m_s must have shape (3,)")
        self.delta_time_s = float(delta_time_s)
        self.mean_chord_m = float(mean_chord_m)
        self.freestream_velocity_m_s = freestream
        self.parameters = parameters
        self._previous_bound_circulation: np.ndarray | None = None
        self._step_index = 0

    def reset_history(self) -> None:
        self._previous_bound_circulation = None
        self._step_index = 0

    @staticmethod
    def _velocity_field(
        value: np.ndarray | Sequence[float] | None,
        shape: tuple[int, int, int],
        default: np.ndarray,
        name: str,
    ) -> np.ndarray:
        if value is None:
            return np.broadcast_to(default, shape).copy()
        array = np.asarray(value, dtype=float)
        if array.shape == (3,):
            return np.broadcast_to(array, shape).copy()
        if array.shape != shape:
            raise ValueError(f"{name} must have shape (3,) or {shape}")
        return array.copy()

    def _wake_velocity(
        self,
        points_m: np.ndarray,
        wake_rings: Sequence[WakeRing],
    ) -> np.ndarray:
        velocity = np.zeros_like(points_m)
        default_initial_core = self.parameters.initial_core_radius_m(self.mean_chord_m)
        for ring in wake_rings:
            corners = np.asarray(ring.corners_m, dtype=float)
            if corners.shape != (4, 3):
                raise ValueError("wake ring corners must have shape (4, 3)")
            initial_core = (
                default_initial_core
                if ring.initial_core_radius_m is None
                else float(ring.initial_core_radius_m)
            )
            core = vortex_core_radius(
                initial_core,
                ring.age_s,
                ring.circulation_m2_s,
                self.parameters.kinematic_viscosity_m2_s,
                lamb_constant=self.parameters.lamb_constant,
                squire_parameter=self.parameters.squire_parameter,
            )
            velocity += vortex_ring_velocity(
                points_m,
                corners,
                ring.circulation_m2_s,
                float(core),
            )
        return velocity

    @staticmethod
    def _local_alpha(
        leading_edge_velocity: np.ndarray,
        leading_edge_chord: np.ndarray,
        leading_edge_normal: np.ndarray,
    ) -> np.ndarray:
        chordwise = np.einsum("ij,ij->i", leading_edge_velocity, leading_edge_chord)
        normal = np.einsum("ij,ij->i", leading_edge_velocity, leading_edge_normal)
        return np.arctan2(-normal, chordwise)

    @staticmethod
    def _build_plev_rings(
        geometry: PanelEdgeGeometry,
        distance_m: np.ndarray,
    ) -> np.ndarray:
        spanwise = geometry.spanwise_panels
        rings = np.empty((spanwise, 4, 3), dtype=float)
        for j in range(spanwise):
            leading_panel = geometry.corners_m[0, j]
            front_left = leading_panel[0]
            front_right = leading_panel[3]
            chord = geometry.chord_directions[0, j]
            if distance_m[j] > geometry.chord_lengths_m[0, j] + 1.0e-12:
                raise ValueError(
                    "d_PLEV exceeds the leading-edge panel chord; the paper gives "
                    "no clipping rule, so this reconstruction fails closed"
                )
            back_left = front_left + distance_m[j] * chord
            back_right = front_right + distance_m[j] * chord
            rings[j] = (front_left, back_left, back_right, front_right)
        return rings

    @staticmethod
    def _ring_aic(
        points: np.ndarray,
        normals: np.ndarray,
        rings: np.ndarray,
        core_radii: np.ndarray,
    ) -> np.ndarray:
        matrix = np.empty((points.shape[0], rings.shape[0]), dtype=float)
        for source in range(rings.shape[0]):
            induced = vortex_ring_velocity(
                points,
                rings[source],
                1.0,
                float(core_radii[source]),
            )
            matrix[:, source] = np.einsum("ij,ij->i", induced, normals)
        return matrix

    def solve_step(
        self,
        vertices_m: np.ndarray,
        *,
        undisturbed_velocity_m_s: np.ndarray | Sequence[float] | None = None,
        surface_velocity_m_s: np.ndarray | Sequence[float] | None = None,
        wake_rings: Sequence[WakeRing] = (),
        alpha_le_rad: np.ndarray | Sequence[float] | None = None,
        enable_plev: bool = True,
    ) -> YangPLEVStepResult:
        """Solve one bound-circulation step with simultaneous PLEV feedback.

        ``undisturbed_velocity_m_s`` is freestream plus any caller-prescribed
        non-vortex flow.  ``surface_velocity_m_s`` is the panel velocity.  Wake
        rings supplied separately are evaluated with Eqs. (8)--(10).  Bound and
        PLEV self-induction are deliberately excluded while defining ``v_LE`` and
        ``alpha_LE`` so Eq. (11) remains a directly auditable coupled linear solve.
        """

        geometry = panel_edge_geometry(vertices_m)
        velocity_shape = geometry.collocation_m.shape
        undisturbed = self._velocity_field(
            undisturbed_velocity_m_s,
            velocity_shape,
            self.freestream_velocity_m_s,
            "undisturbed_velocity_m_s",
        )
        surface = self._velocity_field(
            surface_velocity_m_s,
            velocity_shape,
            np.zeros(3),
            "surface_velocity_m_s",
        )
        wake_velocity = self._wake_velocity(
            geometry.collocation_m.reshape(-1, 3),
            wake_rings,
        ).reshape(velocity_shape)
        relative_velocity = undisturbed + wake_velocity - surface

        leading_velocity = relative_velocity[0]
        if alpha_le_rad is None:
            alpha = self._local_alpha(
                leading_velocity,
                geometry.chord_directions[0],
                geometry.normals[0],
            )
        else:
            alpha = np.asarray(alpha_le_rad, dtype=float)
            if alpha.shape != (geometry.spanwise_panels,):
                raise ValueError(
                    f"alpha_le_rad must have shape ({geometry.spanwise_panels},)"
                )

        local_speed = np.linalg.norm(leading_velocity, axis=1)
        d_plev = 0.25 * local_speed * self.delta_time_s
        r_plev = self.parameters.plev_core_radius_coefficient * d_plev
        plev_rings = self._build_plev_rings(geometry, d_plev)
        k_plev = np.asarray(plev_strength_coefficient(alpha, self.parameters))
        if not enable_plev:
            k_plev = np.zeros_like(k_plev)

        points = geometry.collocation_m.reshape(-1, 3)
        normals = geometry.normals.reshape(-1, 3)
        bound_rings = geometry.corners_m.reshape(-1, 4, 3)
        bound_core = self.parameters.initial_core_radius_m(self.mean_chord_m)
        bound_aic = self._ring_aic(
            points,
            normals,
            bound_rings,
            np.full(bound_rings.shape[0], bound_core),
        )

        # A zero-distance PLEV is a degenerate ring.  Its strength is necessarily
        # zero for a well-defined local AOA, but retain an exactly zero AIC column
        # rather than inventing a core-radius floor.
        plev_unit_aic = np.zeros((points.shape[0], geometry.spanwise_panels))
        active_geometry = d_plev > np.finfo(float).eps
        if np.any(active_geometry):
            active_columns = self._ring_aic(
                points,
                normals,
                plev_rings[active_geometry],
                r_plev[active_geometry],
            )
            plev_unit_aic[:, active_geometry] = active_columns

        coupled_aic = bound_aic.copy()
        for j in range(geometry.spanwise_panels):
            leading_edge_flat_index = j
            coupled_aic[:, leading_edge_flat_index] += k_plev[j] * plev_unit_aic[:, j]

        relative_normal = np.einsum(
            "ijk,ijk->ij",
            relative_velocity,
            geometry.normals,
        ).reshape(-1)
        rhs = -relative_normal
        try:
            bound_flat = np.linalg.solve(coupled_aic, rhs)
        except np.linalg.LinAlgError as error:
            raise np.linalg.LinAlgError(
                "Yang panel-edge coupled AIC is singular for this geometry"
            ) from error
        bound = bound_flat.reshape(
            geometry.chordwise_panels,
            geometry.spanwise_panels,
        )
        gamma_plev = k_plev * bound[0]

        residual = bound_aic @ bound_flat + plev_unit_aic @ gamma_plev + relative_normal
        if self._previous_bound_circulation is None:
            previous = np.zeros_like(bound)
        else:
            if self._previous_bound_circulation.shape != bound.shape:
                raise ValueError("panel topology changed without reset_history()")
            previous = self._previous_bound_circulation.copy()

        result = YangPLEVStepResult(
            step_index=self._step_index,
            delta_time_s=self.delta_time_s,
            geometry=geometry,
            known_relative_velocity_m_s=relative_velocity,
            alpha_le_rad=alpha.copy(),
            k_plev=k_plev.copy(),
            d_plev_m=d_plev.copy(),
            r_plev_m=r_plev.copy(),
            plev_corners_m=plev_rings.copy(),
            bound_circulation_m2_s=bound.copy(),
            plev_circulation_m2_s=gamma_plev.copy(),
            previous_bound_circulation_m2_s=previous,
            bound_aic=bound_aic,
            plev_unit_aic=plev_unit_aic,
            coupled_aic=coupled_aic,
            rhs_normal_velocity_m_s=rhs,
            max_abs_normal_residual_m_s=float(np.max(np.abs(residual))),
            coupled_aic_condition_number=float(np.linalg.cond(coupled_aic)),
        )
        self._previous_bound_circulation = bound.copy()
        self._step_index += 1
        return result

    def solve_history(
        self,
        vertices_history_m: Sequence[np.ndarray],
        *,
        undisturbed_velocity_history_m_s: Sequence[np.ndarray | Sequence[float] | None]
        | None = None,
        surface_velocity_history_m_s: Sequence[np.ndarray | Sequence[float] | None]
        | None = None,
        wake_history: Sequence[Sequence[WakeRing]] | None = None,
        alpha_le_history_rad: Sequence[np.ndarray | Sequence[float] | None]
        | None = None,
        enable_plev: bool = True,
        reset: bool = True,
    ) -> tuple[YangPLEVStepResult, ...]:
        """Run a short externally prescribed rigid-wing geometry history."""

        vertices = list(vertices_history_m)
        count = len(vertices)
        if count == 0:
            return ()
        if reset:
            self.reset_history()

        def normalize_history(
            value: Sequence[object] | None, name: str
        ) -> list[object]:
            if value is None:
                return [None] * count
            normalized = list(value)
            if len(normalized) != count:
                raise ValueError(
                    f"{name} must have the same length as vertices_history_m"
                )
            return normalized

        undisturbed = normalize_history(
            undisturbed_velocity_history_m_s,
            "undisturbed_velocity_history_m_s",
        )
        surface = normalize_history(
            surface_velocity_history_m_s,
            "surface_velocity_history_m_s",
        )
        alpha = normalize_history(alpha_le_history_rad, "alpha_le_history_rad")
        if wake_history is None:
            wakes: list[Sequence[WakeRing]] = [()] * count
        else:
            wakes = list(wake_history)
            if len(wakes) != count:
                raise ValueError(
                    "wake_history must have the same length as vertices_history_m"
                )

        results = []
        for index in range(count):
            results.append(
                self.solve_step(
                    vertices[index],
                    undisturbed_velocity_m_s=undisturbed[index],
                    surface_velocity_m_s=surface[index],
                    wake_rings=wakes[index],
                    alpha_le_rad=alpha[index],
                    enable_plev=enable_plev,
                )
            )
        return tuple(results)


__all__ = (
    "IMPLEMENTATION_CONVENTIONS",
    "MODEL_SCOPE_LIMITATIONS",
    "PanelEdgeGeometry",
    "WakeRing",
    "YANG_2025_PARAMETERS",
    "Yang2025Parameters",
    "YangPLEVSolver",
    "YangPLEVStepResult",
    "adaptive_wake_coefficient",
    "adaptive_wake_split",
    "eddy_viscosity_coefficient",
    "panel_edge_geometry",
    "plev_circulation",
    "plev_strength_coefficient",
    "rectangular_wing_vertices",
    "vortex_core_radius",
    "vortex_ring_velocity",
    "vortex_segment_velocity",
)
