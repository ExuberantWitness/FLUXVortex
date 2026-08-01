"""Finite-angle Xia--Mohseni trailing-edge junction shadow operator.

This module implements only the spatial gate preregistered as
``N2.6e1b2``:

* the existing clockwise :class:`~svi_dw_types.ActualSurface2D` is mapped
  to Xia and Mohseni's counter-clockwise bound-sheet convention;
* the bound sheet is represented by continuous, linearly varying vortex
  panels with independent upper and lower trailing-edge nodal strengths;
* initialization solves no-through plus Kelvin circulation with no forming
  panel; and
* a fixed-wing forming step solves no-through, Kelvin, and Xia--Mohseni
  equation (3.5) in one linear system.

The direction and relative speed of the forming sheet are evaluated from
the *previous* physical trailing-edge state by the already frozen
``finite_angle_sheet_formation`` oracle.  This distinction matters: the
mathematical endpoint of a finite-angle panel discretization is a corner
singularity and is not silently promoted to a time-continuous formation
provider.

There is deliberately no pressure, force, target response, fitted
coefficient, viscous closure, vortex core, wake roll-up, or production
coupling here.  The trailing-edge crop is an explicit numerical refinement
coordinate supplied by the caller, not a model constant.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .finite_angle_sheet_formation import (
    FiniteAngleSheetFormation,
    finite_angle_sheet_formation,
)
from .svi_dw_types import ActualSurface2D, SVIDWValidationError


_TWO_PI = 2.0 * np.pi


class XMJunctionError(RuntimeError):
    """The shadow junction system is singular or physically inadmissible."""


def _readonly_array(
    name: str,
    value: Any,
    *,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    array = np.array(value, dtype=float, copy=True)
    if shape is not None and array.shape != shape:
        raise SVIDWValidationError(
            f"{name} must have shape {shape}, got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise SVIDWValidationError(f"{name} contains non-finite values")
    array.setflags(write=False)
    return array


def _finite_scalar(name: str, value: Any) -> float:
    scalar = float(value)
    if not np.isfinite(scalar):
        raise SVIDWValidationError(f"{name} must be finite")
    return scalar


def _positive_scalar(name: str, value: Any) -> float:
    scalar = _finite_scalar(name, value)
    if scalar <= 0.0:
        raise SVIDWValidationError(f"{name} must be positive")
    return scalar


def _unit_vector(name: str, value: Any) -> np.ndarray:
    vector = _readonly_array(name, value, shape=(2,))
    norm = float(np.linalg.norm(vector))
    if abs(norm - 1.0) > 2.0e-12:
        raise SVIDWValidationError(f"{name} must be a unit vector")
    return vector


def _rotate_ccw(vector: np.ndarray, angle_rad: float) -> np.ndarray:
    cosine = np.cos(angle_rad)
    sine = np.sin(angle_rad)
    result = np.array(
        (
            cosine * vector[0] - sine * vector[1],
            sine * vector[0] + cosine * vector[1],
        ),
        dtype=float,
    )
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class LinearVortexPanelVelocityBasis2D:
    """Velocity per unit start/end nodal sheet strength.

    Arrays have shape ``(n_target, n_panel, 2)``.  Positive sheet strength
    denotes positive (counter-clockwise) point-vortex circulation per unit
    arclength; it is independent of the order in which an equivalent
    physical panel is parameterized.
    """

    start_node_velocity: np.ndarray
    end_node_velocity: np.ndarray

    def __post_init__(self) -> None:
        start = _readonly_array(
            "start_node_velocity", self.start_node_velocity
        )
        end = _readonly_array("end_node_velocity", self.end_node_velocity)
        if start.ndim != 3 or start.shape[-1] != 2:
            raise SVIDWValidationError(
                "start_node_velocity must have shape (n_target,n_panel,2)"
            )
        if end.shape != start.shape:
            raise SVIDWValidationError(
                "end_node_velocity must match start_node_velocity"
            )
        object.__setattr__(self, "start_node_velocity", start)
        object.__setattr__(self, "end_node_velocity", end)


@dataclass(frozen=True)
class XMACroppedGeometry2D:
    """Counter-clockwise actual wall with an explicit symmetric TE crop."""

    source_surface: ActualSurface2D
    contour_nodes_ccw: np.ndarray
    panel_midpoints: np.ndarray
    panel_tangents: np.ndarray
    panel_outward_normals: np.ndarray
    panel_lengths: np.ndarray
    trailing_edge_body: np.ndarray
    upper_outgoing_tangent: np.ndarray
    lower_outgoing_tangent: np.ndarray
    trailing_edge_wedge_angle_rad: float
    trailing_edge_reference_panel_length: float
    epsilon: float
    epsilon_over_te_panel: float

    def __post_init__(self) -> None:
        if not isinstance(self.source_surface, ActualSurface2D):
            raise SVIDWValidationError(
                "source_surface must be an ActualSurface2D"
            )
        n_panel = self.source_surface.panel_count
        nodes = _readonly_array(
            "contour_nodes_ccw",
            self.contour_nodes_ccw,
            shape=(n_panel + 1, 2),
        )
        midpoints = _readonly_array(
            "panel_midpoints",
            self.panel_midpoints,
            shape=(n_panel, 2),
        )
        tangents = _readonly_array(
            "panel_tangents",
            self.panel_tangents,
            shape=(n_panel, 2),
        )
        normals = _readonly_array(
            "panel_outward_normals",
            self.panel_outward_normals,
            shape=(n_panel, 2),
        )
        lengths = _readonly_array(
            "panel_lengths", self.panel_lengths, shape=(n_panel,)
        )
        trailing_edge = _readonly_array(
            "trailing_edge_body", self.trailing_edge_body, shape=(2,)
        )
        upper = _unit_vector(
            "upper_outgoing_tangent", self.upper_outgoing_tangent
        )
        lower = _unit_vector(
            "lower_outgoing_tangent", self.lower_outgoing_tangent
        )
        wedge = _positive_scalar(
            "trailing_edge_wedge_angle_rad",
            self.trailing_edge_wedge_angle_rad,
        )
        reference_length = _positive_scalar(
            "trailing_edge_reference_panel_length",
            self.trailing_edge_reference_panel_length,
        )
        epsilon = _positive_scalar("epsilon", self.epsilon)
        ratio = _positive_scalar(
            "epsilon_over_te_panel", self.epsilon_over_te_panel
        )
        if wedge >= np.pi:
            raise SVIDWValidationError(
                "trailing-edge wedge angle must be less than pi"
            )
        if epsilon >= reference_length or ratio >= 1.0:
            raise SVIDWValidationError(
                "epsilon must crop less than one trailing-edge panel"
            )
        if np.any(lengths <= 0.0):
            raise SVIDWValidationError(
                "cropped geometry contains a non-positive panel length"
            )
        if abs(epsilon - ratio * reference_length) > (
            1024.0
            * np.finfo(float).eps
            * max(epsilon, reference_length, 1.0)
        ):
            raise SVIDWValidationError(
                "epsilon is inconsistent with epsilon_over_te_panel"
            )
        if np.linalg.norm(nodes[0] - trailing_edge) <= epsilon * 0.5:
            raise SVIDWValidationError(
                "upper bound sheet was not cropped away from the TE"
            )
        if np.linalg.norm(nodes[-1] - trailing_edge) <= epsilon * 0.5:
            raise SVIDWValidationError(
                "lower bound sheet was not cropped away from the TE"
            )
        object.__setattr__(self, "contour_nodes_ccw", nodes)
        object.__setattr__(self, "panel_midpoints", midpoints)
        object.__setattr__(self, "panel_tangents", tangents)
        object.__setattr__(self, "panel_outward_normals", normals)
        object.__setattr__(self, "panel_lengths", lengths)
        object.__setattr__(self, "trailing_edge_body", trailing_edge)
        object.__setattr__(self, "upper_outgoing_tangent", upper)
        object.__setattr__(self, "lower_outgoing_tangent", lower)
        object.__setattr__(
            self, "trailing_edge_wedge_angle_rad", wedge
        )
        object.__setattr__(
            self, "trailing_edge_reference_panel_length", reference_length
        )
        object.__setattr__(self, "epsilon", epsilon)
        object.__setattr__(self, "epsilon_over_te_panel", ratio)

    @property
    def panel_count(self) -> int:
        return int(self.panel_lengths.size)

    @property
    def chord(self) -> float:
        return float(self.source_surface.section.chord)

    @property
    def trailing_edge_wedge_angle_deg(self) -> float:
        return float(np.rad2deg(self.trailing_edge_wedge_angle_rad))


@dataclass(frozen=True)
class XMCirculationLedger2D:
    """One explicitly closed circulation account."""

    bound_circulation_ccw: float
    forming_circulation_ccw: float
    historical_wake_circulation_ccw: float
    total_circulation_ccw: float
    prescribed_total_circulation_ccw: float
    kelvin_residual: float
    normalized_kelvin_residual: float


@dataclass(frozen=True)
class XMResidualDiagnostics2D:
    """Named equation residuals for the coupled linear solve."""

    normal_velocity_residual: np.ndarray
    maximum_relative_normal_residual: float
    kelvin_residual: float
    normalized_kelvin_residual: float
    kutta_residual: float | None
    normalized_kutta_residual: float | None
    linear_system_residual: np.ndarray
    maximum_normalized_linear_system_residual: float
    system_condition_number: float

    def __post_init__(self) -> None:
        normal = _readonly_array(
            "normal_velocity_residual", self.normal_velocity_residual
        )
        linear = _readonly_array(
            "linear_system_residual", self.linear_system_residual
        )
        if normal.ndim != 1 or linear.ndim != 1:
            raise SVIDWValidationError(
                "residual vectors must be one-dimensional"
            )
        finite_values = (
            self.maximum_relative_normal_residual,
            self.kelvin_residual,
            self.normalized_kelvin_residual,
            self.maximum_normalized_linear_system_residual,
            self.system_condition_number,
        )
        if not all(np.isfinite(float(item)) for item in finite_values):
            raise SVIDWValidationError(
                "residual diagnostics contain non-finite values"
            )
        for name, value in (
            ("kutta_residual", self.kutta_residual),
            ("normalized_kutta_residual", self.normalized_kutta_residual),
        ):
            if value is not None and not np.isfinite(float(value)):
                raise SVIDWValidationError(f"{name} must be finite or None")
        object.__setattr__(self, "normal_velocity_residual", normal)
        object.__setattr__(self, "linear_system_residual", linear)


@dataclass(frozen=True)
class XMJunctionState2D:
    """Auditable initialization or one fixed-wing forming-panel state."""

    stage: str
    geometry: XMACroppedGeometry2D
    freestream_velocity_body: np.ndarray
    bound_node_strength_ccw: np.ndarray
    gamma1_upper_physical: float
    gamma2_lower_physical: float
    u1_plus: float
    u2_minus: float
    previous_u1_plus: float | None
    previous_u2_minus: float | None
    formation: FiniteAngleSheetFormation | None
    forming_direction_body: np.ndarray | None
    forming_start_body: np.ndarray | None
    forming_end_body: np.ndarray | None
    forming_sheet_strength_ccw: float
    forming_length: float
    no_birth: bool
    circulation: XMCirculationLedger2D
    residuals: XMResidualDiagnostics2D

    def __post_init__(self) -> None:
        if self.stage not in {"initialization", "forming", "no_birth"}:
            raise SVIDWValidationError(
                "stage must be initialization, forming, or no_birth"
            )
        if not isinstance(self.geometry, XMACroppedGeometry2D):
            raise SVIDWValidationError(
                "geometry must be an XMACroppedGeometry2D"
            )
        velocity = _readonly_array(
            "freestream_velocity_body",
            self.freestream_velocity_body,
            shape=(2,),
        )
        strengths = _readonly_array(
            "bound_node_strength_ccw",
            self.bound_node_strength_ccw,
            shape=(self.geometry.panel_count + 1,),
        )
        for name, value in (
            ("gamma1_upper_physical", self.gamma1_upper_physical),
            ("gamma2_lower_physical", self.gamma2_lower_physical),
            ("u1_plus", self.u1_plus),
            ("u2_minus", self.u2_minus),
            (
                "forming_sheet_strength_ccw",
                self.forming_sheet_strength_ccw,
            ),
            ("forming_length", self.forming_length),
        ):
            _finite_scalar(name, value)
        if not isinstance(self.no_birth, (bool, np.bool_)):
            raise SVIDWValidationError("no_birth must be boolean")
        optional_vectors = (
            ("forming_direction_body", self.forming_direction_body),
            ("forming_start_body", self.forming_start_body),
            ("forming_end_body", self.forming_end_body),
        )
        normalized_optional: dict[str, np.ndarray | None] = {}
        for name, value in optional_vectors:
            normalized_optional[name] = (
                None
                if value is None
                else _readonly_array(name, value, shape=(2,))
            )
        if self.no_birth:
            if self.formation is not None or any(
                value is not None for _, value in optional_vectors
            ):
                raise SVIDWValidationError(
                    "no-birth state must not fabricate formation geometry"
                )
            if (
                self.forming_sheet_strength_ccw != 0.0
                or self.forming_length != 0.0
            ):
                raise SVIDWValidationError(
                    "no-birth state must have zero forming sheet"
                )
        elif self.stage == "forming":
            if not isinstance(self.formation, FiniteAngleSheetFormation):
                raise SVIDWValidationError(
                    "forming stage requires a formation oracle result"
                )
            if any(value is None for _, value in optional_vectors):
                raise SVIDWValidationError(
                    "forming stage requires direction and endpoints"
                )
        if not isinstance(self.circulation, XMCirculationLedger2D):
            raise SVIDWValidationError(
                "circulation must be an XMCirculationLedger2D"
            )
        if not isinstance(self.residuals, XMResidualDiagnostics2D):
            raise SVIDWValidationError(
                "residuals must be an XMResidualDiagnostics2D"
            )
        object.__setattr__(self, "freestream_velocity_body", velocity)
        object.__setattr__(self, "bound_node_strength_ccw", strengths)
        for name, value in normalized_optional.items():
            object.__setattr__(self, name, value)


def build_xia_ccw_geometry(
    surface: ActualSurface2D,
    *,
    epsilon_over_te_panel: float,
) -> XMACroppedGeometry2D:
    """Map the actual wall to Xia's CCW convention and crop both TE arms."""
    if not isinstance(surface, ActualSurface2D):
        raise SVIDWValidationError(
            "surface must be an ActualSurface2D"
        )
    ratio = _positive_scalar(
        "epsilon_over_te_panel", epsilon_over_te_panel
    )
    if ratio >= 1.0:
        raise SVIDWValidationError(
            "epsilon_over_te_panel must lie in (0,1)"
        )

    # ActualSurface2D is clockwise lower-TE -> LE -> upper-TE.  Reversal is
    # Xia's counter-clockwise upper-TE -> LE -> lower-TE convention.
    original = np.array(surface.contour_nodes[::-1], dtype=float, copy=True)
    trailing_edge = 0.5 * (original[0] + original[-1])
    upper_te_length = float(np.linalg.norm(original[1] - trailing_edge))
    lower_te_length = float(np.linalg.norm(original[-2] - trailing_edge))
    reference_length = min(upper_te_length, lower_te_length)
    epsilon = ratio * reference_length
    original[0] = (
        trailing_edge
        + epsilon * (original[1] - trailing_edge) / upper_te_length
    )
    original[-1] = (
        trailing_edge
        + epsilon * (original[-2] - trailing_edge) / lower_te_length
    )

    segment = np.diff(original, axis=0)
    lengths = np.linalg.norm(segment, axis=1)
    if np.any(lengths <= 0.0):
        raise SVIDWValidationError(
            "TE crop produced a non-positive panel length"
        )
    tangents = segment / lengths[:, None]
    # For a CCW body contour, the exterior is on the right.
    outward = np.column_stack((tangents[:, 1], -tangents[:, 0]))
    midpoints = 0.5 * (original[:-1] + original[1:])

    upper_outgoing = -tangents[0]
    lower_outgoing = tangents[-1]
    signed_wedge = float(
        np.arctan2(
            upper_outgoing[0] * lower_outgoing[1]
            - upper_outgoing[1] * lower_outgoing[0],
            np.dot(upper_outgoing, lower_outgoing),
        )
    )
    if signed_wedge <= 0.0:
        signed_wedge += _TWO_PI
    if not (0.0 < signed_wedge < np.pi):
        raise SVIDWValidationError(
            "actual trailing edge does not define a finite downstream wedge"
        )
    return XMACroppedGeometry2D(
        source_surface=surface,
        contour_nodes_ccw=original,
        panel_midpoints=midpoints,
        panel_tangents=tangents,
        panel_outward_normals=outward,
        panel_lengths=lengths,
        trailing_edge_body=trailing_edge,
        upper_outgoing_tangent=upper_outgoing,
        lower_outgoing_tangent=lower_outgoing,
        trailing_edge_wedge_angle_rad=signed_wedge,
        trailing_edge_reference_panel_length=reference_length,
        epsilon=epsilon,
        epsilon_over_te_panel=ratio,
    )


def linear_vortex_panel_velocity_basis(
    points: Any,
    panel_starts: Any,
    panel_ends: Any,
) -> LinearVortexPanelVelocityBasis2D:
    """Return analytic velocities of both linear nodal panel bases.

    The formulas are exact line integrals of the positive-CCW point-vortex
    kernel.  Targets may approach a panel from either side but may not
    coincide with a panel endpoint.  Boundary collocation principal values
    are installed separately by the bound-system assembler.
    """
    targets = np.asarray(points, dtype=float)
    starts = np.asarray(panel_starts, dtype=float)
    ends = np.asarray(panel_ends, dtype=float)
    if (
        targets.ndim != 2
        or targets.shape[1:] != (2,)
        or not np.all(np.isfinite(targets))
    ):
        raise SVIDWValidationError(
            "points must be finite with shape (n_target,2)"
        )
    if (
        starts.ndim != 2
        or starts.shape[1:] != (2,)
        or not np.all(np.isfinite(starts))
    ):
        raise SVIDWValidationError(
            "panel_starts must be finite with shape (n_panel,2)"
        )
    if ends.shape != starts.shape or not np.all(np.isfinite(ends)):
        raise SVIDWValidationError(
            "panel_ends must match panel_starts and be finite"
        )
    segment = ends - starts
    length = np.linalg.norm(segment, axis=1)
    if np.any(length <= 0.0):
        raise SVIDWValidationError(
            "linear vortex panels must have positive length"
        )
    tangent = segment / length[:, None]
    left_normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))

    relative = targets[:, None, :] - starts[None, :, :]
    x_local = np.einsum("mpj,pj->mp", relative, tangent)
    y_local = np.einsum("mpj,pj->mp", relative, left_normal)
    x_after = x_local - length[None, :]
    radius_start_sq = x_local**2 + y_local**2
    radius_end_sq = x_after**2 + y_local**2
    coordinate_scale = max(
        float(np.max(length, initial=0.0)),
        float(np.max(np.abs(targets), initial=0.0)),
        float(np.max(np.abs(starts), initial=0.0)),
        float(np.max(np.abs(ends), initial=0.0)),
        1.0,
    )
    endpoint_tolerance_sq = (
        256.0 * np.finfo(float).eps * coordinate_scale
    ) ** 2
    if np.any(radius_start_sq <= endpoint_tolerance_sq) or np.any(
        radius_end_sq <= endpoint_tolerance_sq
    ):
        raise SVIDWValidationError(
            "a velocity target coincides with a vortex-panel endpoint"
        )

    angle = np.arctan2(
        y_local * length[None, :],
        x_local * x_after + y_local**2,
    )
    log_ratio = 0.5 * np.log(radius_start_sq / radius_end_sq)
    first_angle_moment = x_local * angle - y_local * log_ratio
    first_log_moment = (
        x_local * log_ratio
        - length[None, :]
        + y_local * angle
    )

    start_tangent = -(
        angle - first_angle_moment / length[None, :]
    ) / _TWO_PI
    start_normal = (
        log_ratio - first_log_moment / length[None, :]
    ) / _TWO_PI
    end_tangent = -(
        first_angle_moment / length[None, :]
    ) / _TWO_PI
    end_normal = (
        first_log_moment / length[None, :]
    ) / _TWO_PI
    start_velocity = (
        start_tangent[..., None] * tangent[None, :, :]
        + start_normal[..., None] * left_normal[None, :, :]
    )
    end_velocity = (
        end_tangent[..., None] * tangent[None, :, :]
        + end_normal[..., None] * left_normal[None, :, :]
    )
    return LinearVortexPanelVelocityBasis2D(
        start_node_velocity=start_velocity,
        end_node_velocity=end_velocity,
    )


def _bound_velocity_basis_at_collocation(
    geometry: XMACroppedGeometry2D,
) -> tuple[np.ndarray, np.ndarray]:
    n_panel = geometry.panel_count
    raw = linear_vortex_panel_velocity_basis(
        geometry.panel_midpoints,
        geometry.contour_nodes_ccw[:-1],
        geometry.contour_nodes_ccw[1:],
    )
    start = np.array(raw.start_node_velocity, copy=True)
    end = np.array(raw.end_node_velocity, copy=True)

    # Principal-value self influence for a linear vortex panel at its
    # midpoint.  The average tangential trace is zero; the two nodal normal
    # bases are +1/(2pi) and -1/(2pi) along the panel's left normal.
    diagonal = np.arange(n_panel)
    left_normal = np.column_stack(
        (
            -geometry.panel_tangents[:, 1],
            geometry.panel_tangents[:, 0],
        )
    )
    start[diagonal, diagonal] = left_normal / _TWO_PI
    end[diagonal, diagonal] = -left_normal / _TWO_PI

    node_velocity = np.zeros((n_panel, n_panel + 1, 2), dtype=float)
    node_velocity[:, :-1] += start
    node_velocity[:, 1:] += end
    normal = np.einsum(
        "mnj,mj->mn",
        node_velocity,
        geometry.panel_outward_normals,
    )
    return node_velocity, normal


def _bound_circulation_weights(
    geometry: XMACroppedGeometry2D,
) -> np.ndarray:
    weights = np.zeros(geometry.panel_count + 1, dtype=float)
    weights[:-1] += 0.5 * geometry.panel_lengths
    weights[1:] += 0.5 * geometry.panel_lengths
    return weights


def _forming_unit_velocity(
    points: np.ndarray,
    *,
    start: np.ndarray,
    end: np.ndarray,
) -> np.ndarray:
    basis = linear_vortex_panel_velocity_basis(
        points, start[None, :], end[None, :]
    )
    # Equal endpoint strengths are a constant-strength vortex panel.
    return (
        basis.start_node_velocity[:, 0, :]
        + basis.end_node_velocity[:, 0, :]
    )


def _solve_linear_system(
    system: np.ndarray,
    rhs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    condition_number = float(np.linalg.cond(system))
    if not np.isfinite(condition_number):
        raise XMJunctionError(
            "Xia--Mohseni junction matrix has non-finite condition number"
        )
    try:
        solution = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError as error:
        raise XMJunctionError(
            "Xia--Mohseni junction matrix is singular"
        ) from error
    residual = system @ solution - rhs
    if not np.all(np.isfinite(solution)) or not np.all(
        np.isfinite(residual)
    ):
        raise XMJunctionError(
            "Xia--Mohseni junction solve produced non-finite values"
        )
    return solution, residual, condition_number


def _build_diagnostics(
    *,
    geometry: XMACroppedGeometry2D,
    freestream_speed: float,
    normal_residual: np.ndarray,
    kelvin_residual: float,
    kutta_residual: float | None,
    linear_residual: np.ndarray,
    condition_number: float,
) -> XMResidualDiagnostics2D:
    speed_scale = max(freestream_speed, np.finfo(float).tiny)
    circulation_scale = max(
        speed_scale * geometry.chord, np.finfo(float).tiny
    )
    normalized_blocks = [
        np.abs(linear_residual[: geometry.panel_count]) / speed_scale,
        np.array(
            [
                abs(linear_residual[geometry.panel_count])
                / circulation_scale
            ]
        ),
    ]
    if linear_residual.size == geometry.panel_count + 2:
        normalized_blocks.append(
            np.array([abs(linear_residual[-1]) / speed_scale])
        )
    maximum_linear = float(
        np.max(np.concatenate(normalized_blocks), initial=0.0)
    )
    return XMResidualDiagnostics2D(
        normal_velocity_residual=normal_residual,
        maximum_relative_normal_residual=float(
            np.max(np.abs(normal_residual), initial=0.0) / speed_scale
        ),
        kelvin_residual=float(kelvin_residual),
        normalized_kelvin_residual=float(
            abs(kelvin_residual) / circulation_scale
        ),
        kutta_residual=(
            None if kutta_residual is None else float(kutta_residual)
        ),
        normalized_kutta_residual=(
            None
            if kutta_residual is None
            else float(abs(kutta_residual) / speed_scale)
        ),
        linear_system_residual=linear_residual,
        maximum_normalized_linear_system_residual=maximum_linear,
        system_condition_number=condition_number,
    )


def _state_from_solution(
    *,
    stage: str,
    geometry: XMACroppedGeometry2D,
    freestream: np.ndarray,
    bound_strength: np.ndarray,
    formation: FiniteAngleSheetFormation | None,
    previous_u1_plus: float | None,
    previous_u2_minus: float | None,
    forming_direction: np.ndarray | None,
    forming_start: np.ndarray | None,
    forming_end: np.ndarray | None,
    forming_strength: float,
    no_birth: bool,
    node_normal_matrix: np.ndarray,
    forming_normal: np.ndarray | None,
    linear_residual: np.ndarray,
    condition_number: float,
) -> XMJunctionState2D:
    weights = _bound_circulation_weights(geometry)
    bound_circulation = float(weights @ bound_strength)
    forming_length = (
        0.0
        if forming_start is None or forming_end is None
        else float(np.linalg.norm(forming_end - forming_start))
    )
    forming_circulation = float(forming_strength * forming_length)
    total_circulation = bound_circulation + forming_circulation
    kelvin_residual = total_circulation
    normal_residual = (
        node_normal_matrix @ bound_strength
        + geometry.panel_outward_normals @ freestream
    )
    if forming_normal is not None:
        normal_residual = (
            normal_residual + forming_normal * forming_strength
        )
    kutta_residual: float | None
    if formation is None:
        kutta_residual = None
    else:
        kutta_residual = float(
            forming_strength
            - bound_strength[0] * np.cos(formation.delta_theta1)
            - bound_strength[-1] * np.cos(formation.delta_theta2)
        )
    speed = float(np.linalg.norm(freestream))
    diagnostics = _build_diagnostics(
        geometry=geometry,
        freestream_speed=speed,
        normal_residual=normal_residual,
        kelvin_residual=kelvin_residual,
        kutta_residual=kutta_residual,
        linear_residual=linear_residual,
        condition_number=condition_number,
    )
    ledger = XMCirculationLedger2D(
        bound_circulation_ccw=bound_circulation,
        forming_circulation_ccw=forming_circulation,
        historical_wake_circulation_ccw=0.0,
        total_circulation_ccw=total_circulation,
        prescribed_total_circulation_ccw=0.0,
        kelvin_residual=kelvin_residual,
        normalized_kelvin_residual=diagnostics.normalized_kelvin_residual,
    )
    gamma1 = float(bound_strength[0])
    gamma2 = float(bound_strength[-1])
    return XMJunctionState2D(
        stage=stage,
        geometry=geometry,
        freestream_velocity_body=freestream,
        bound_node_strength_ccw=bound_strength,
        gamma1_upper_physical=gamma1,
        gamma2_lower_physical=gamma2,
        u1_plus=gamma1,
        u2_minus=-gamma2,
        previous_u1_plus=previous_u1_plus,
        previous_u2_minus=previous_u2_minus,
        formation=formation,
        forming_direction_body=forming_direction,
        forming_start_body=forming_start,
        forming_end_body=forming_end,
        forming_sheet_strength_ccw=forming_strength,
        forming_length=forming_length,
        no_birth=no_birth,
        circulation=ledger,
        residuals=diagnostics,
    )


def initialize_xm_bound_state(
    geometry: XMACroppedGeometry2D,
    *,
    freestream_velocity_body: Any,
) -> XMJunctionState2D:
    """Solve the no-forming-panel ``N normal + Kelvin`` initialization."""
    if not isinstance(geometry, XMACroppedGeometry2D):
        raise SVIDWValidationError(
            "geometry must be an XMACroppedGeometry2D"
        )
    freestream = _readonly_array(
        "freestream_velocity_body",
        freestream_velocity_body,
        shape=(2,),
    )
    speed = float(np.linalg.norm(freestream))
    if speed <= 0.0:
        raise SVIDWValidationError(
            "freestream_velocity_body must have positive magnitude"
        )
    _, node_normal = _bound_velocity_basis_at_collocation(geometry)
    weights = _bound_circulation_weights(geometry)
    n_panel = geometry.panel_count
    system = np.empty((n_panel + 1, n_panel + 1), dtype=float)
    rhs = np.empty(n_panel + 1, dtype=float)
    system[:n_panel] = node_normal
    rhs[:n_panel] = -(
        geometry.panel_outward_normals @ freestream
    )
    system[-1] = weights
    rhs[-1] = 0.0
    solution, linear_residual, condition_number = _solve_linear_system(
        system, rhs
    )
    no_birth = bool(
        geometry.source_surface.section.maximum_camber == 0.0
        and freestream[0] > 0.0
        and freestream[1] == 0.0
    )
    return _state_from_solution(
        stage="no_birth" if no_birth else "initialization",
        geometry=geometry,
        freestream=freestream,
        bound_strength=solution,
        formation=None,
        previous_u1_plus=None,
        previous_u2_minus=None,
        forming_direction=None,
        forming_start=None,
        forming_end=None,
        forming_strength=0.0,
        no_birth=no_birth,
        node_normal_matrix=node_normal,
        forming_normal=None,
        linear_residual=linear_residual,
        condition_number=condition_number,
    )


def solve_xm_forming_step(
    geometry: XMACroppedGeometry2D,
    *,
    freestream_velocity_body: Any,
    previous_u1_plus: float,
    previous_u2_minus: float,
    time_step: float,
) -> XMJunctionState2D:
    """Solve one fixed-wing finite-angle forming-panel spatial step.

    ``previous_u1_plus`` and ``previous_u2_minus`` are the two physical
    incident-side velocities in Xia--Mohseni's convention.  They determine
    only this step's direction and length.  The current bound nodal
    strengths and forming strength are then solved simultaneously.
    """
    if not isinstance(geometry, XMACroppedGeometry2D):
        raise SVIDWValidationError(
            "geometry must be an XMACroppedGeometry2D"
        )
    freestream = _readonly_array(
        "freestream_velocity_body",
        freestream_velocity_body,
        shape=(2,),
    )
    speed = float(np.linalg.norm(freestream))
    if speed <= 0.0:
        raise SVIDWValidationError(
            "freestream_velocity_body must have positive magnitude"
        )
    previous_first = _finite_scalar(
        "previous_u1_plus", previous_u1_plus
    )
    previous_second = _finite_scalar(
        "previous_u2_minus", previous_u2_minus
    )
    dt = _positive_scalar("time_step", time_step)

    # A symmetric zero incident state is a genuine no-birth state.  Calling
    # the finite-angle oracle would otherwise invent a direction from 0/0.
    if previous_first == 0.0 and previous_second == 0.0:
        initialized = initialize_xm_bound_state(
            geometry, freestream_velocity_body=freestream
        )
        return XMJunctionState2D(
            stage="no_birth",
            geometry=initialized.geometry,
            freestream_velocity_body=initialized.freestream_velocity_body,
            bound_node_strength_ccw=initialized.bound_node_strength_ccw,
            gamma1_upper_physical=initialized.gamma1_upper_physical,
            gamma2_lower_physical=initialized.gamma2_lower_physical,
            u1_plus=initialized.u1_plus,
            u2_minus=initialized.u2_minus,
            previous_u1_plus=previous_first,
            previous_u2_minus=previous_second,
            formation=None,
            forming_direction_body=None,
            forming_start_body=None,
            forming_end_body=None,
            forming_sheet_strength_ccw=0.0,
            forming_length=0.0,
            no_birth=True,
            circulation=initialized.circulation,
            residuals=initialized.residuals,
        )

    formation = finite_angle_sheet_formation(
        u1_plus=previous_first,
        u2_minus=previous_second,
        wedge_angle_deg=geometry.trailing_edge_wedge_angle_deg,
    )
    # Equal non-zero incident streams are the steady finite-angle limit:
    # gamma_g = dot(Gamma_g) = 0 and the oracle intentionally leaves u_g
    # unidentified.  This is no birth, not a license to choose a direction.
    if not formation.state_identifiable:
        initialized = initialize_xm_bound_state(
            geometry, freestream_velocity_body=freestream
        )
        return XMJunctionState2D(
            stage="no_birth",
            geometry=initialized.geometry,
            freestream_velocity_body=initialized.freestream_velocity_body,
            bound_node_strength_ccw=initialized.bound_node_strength_ccw,
            gamma1_upper_physical=initialized.gamma1_upper_physical,
            gamma2_lower_physical=initialized.gamma2_lower_physical,
            u1_plus=initialized.u1_plus,
            u2_minus=initialized.u2_minus,
            previous_u1_plus=previous_first,
            previous_u2_minus=previous_second,
            formation=None,
            forming_direction_body=None,
            forming_start_body=None,
            forming_end_body=None,
            forming_sheet_strength_ccw=0.0,
            forming_length=0.0,
            no_birth=True,
            circulation=initialized.circulation,
            residuals=initialized.residuals,
        )
    if (
        formation.relative_velocity is None
        or formation.relative_velocity <= 0.0
    ):
        raise XMJunctionError(
            "previous TE state does not identify a positive forming speed"
        )
    direction = _rotate_ccw(
        geometry.upper_outgoing_tangent,
        formation.delta_theta1,
    )
    forming_start = (
        geometry.trailing_edge_body + geometry.epsilon * direction
    )
    forming_length = formation.relative_velocity * dt
    forming_end = forming_start + forming_length * direction

    _, node_normal = _bound_velocity_basis_at_collocation(geometry)
    forming_velocity = _forming_unit_velocity(
        geometry.panel_midpoints,
        start=forming_start,
        end=forming_end,
    )
    forming_normal = np.einsum(
        "ij,ij->i",
        forming_velocity,
        geometry.panel_outward_normals,
    )
    weights = _bound_circulation_weights(geometry)
    n_panel = geometry.panel_count
    system = np.zeros((n_panel + 2, n_panel + 2), dtype=float)
    rhs = np.zeros(n_panel + 2, dtype=float)
    system[:n_panel, : n_panel + 1] = node_normal
    system[:n_panel, -1] = forming_normal
    rhs[:n_panel] = -(
        geometry.panel_outward_normals @ freestream
    )
    system[n_panel, : n_panel + 1] = weights
    system[n_panel, -1] = forming_length
    # Xia--Mohseni (2017), published Eq. (3.5):
    # gamma_g = gamma_1 cos(delta_theta_1)
    #         + gamma_2 cos(delta_theta_2).
    system[-1, 0] = -np.cos(formation.delta_theta1)
    system[-1, n_panel] = -np.cos(formation.delta_theta2)
    system[-1, -1] = 1.0

    solution, linear_residual, condition_number = _solve_linear_system(
        system, rhs
    )
    return _state_from_solution(
        stage="forming",
        geometry=geometry,
        freestream=freestream,
        bound_strength=solution[:-1],
        formation=formation,
        previous_u1_plus=previous_first,
        previous_u2_minus=previous_second,
        forming_direction=direction,
        forming_start=forming_start,
        forming_end=forming_end,
        forming_strength=float(solution[-1]),
        no_birth=False,
        node_normal_matrix=node_normal,
        forming_normal=forming_normal,
        linear_residual=linear_residual,
        condition_number=condition_number,
    )
