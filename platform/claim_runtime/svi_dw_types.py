"""Typed state and actual-surface geometry for the N2.6e1 SVI-DW shadow.

This module is the *S0 foundation* of ``N2.6e1``.  It deliberately contains
no target-force field, force multiplier, fitted force offset, boundary-layer
closure, or wake evolution law.  Its purpose is narrower:

* construct a closed, two-sided NACA four-digit wall in two dimensions;
* give later strong viscous--inviscid coupling an explicitly typed IBL state;
* distinguish the trailing-edge and separated-shear-layer wake branches; and
* derive wall traction from pressure and shear rather than accepting a target
  force as an input.

The active IBL and double-wake equations are intentionally absent at S0.
``svi_dw_outer_2d`` accepts only the quiescent versions of these states until
those equations pass their own source-method gates.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
from typing import Any, Mapping

import numpy as np


class SVIDWValidationError(ValueError):
    """A typed SVI-DW configuration, geometry, or state is invalid."""


class SVIDWFoundationScopeError(NotImplementedError):
    """A caller requested physics that is outside the N2.6e1 S0 foundation."""


class IBLRegime(str, Enum):
    """Equation set active at one canonical boundary-layer station."""

    INACTIVE = "inactive"
    LAMINAR = "laminar"
    TURBULENT = "turbulent"


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


def _integer_at_least(name: str, value: Any, minimum: int) -> int:
    integer = int(value)
    if integer != value or integer < minimum:
        raise SVIDWValidationError(
            f"{name} must be an integer >= {minimum}"
        )
    return integer


def _readonly_array(
    name: str,
    value: Any,
    *,
    shape: tuple[int, ...] | None = None,
    ndim: int | None = None,
    dtype: Any = float,
) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    if shape is not None and array.shape != shape:
        raise SVIDWValidationError(
            f"{name} must have shape {shape}, got {array.shape}"
        )
    if ndim is not None and array.ndim != ndim:
        raise SVIDWValidationError(
            f"{name} must have ndim={ndim}, got {array.shape}"
        )
    if np.issubdtype(array.dtype, np.number) and not np.all(
        np.isfinite(array)
    ):
        raise SVIDWValidationError(f"{name} contains non-finite values")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class NACA4SectionConfig:
    """Geometry-only NACA four-digit section definition."""

    maximum_camber: float = 0.0
    camber_location: float = 0.4
    thickness_ratio: float = 0.15
    chord: float = 1.0
    closed_trailing_edge: bool = True

    def __post_init__(self) -> None:
        m = _finite_scalar("maximum_camber", self.maximum_camber)
        p = _finite_scalar("camber_location", self.camber_location)
        thickness = _positive_scalar(
            "thickness_ratio", self.thickness_ratio
        )
        chord = _positive_scalar("chord", self.chord)
        if m < 0.0 or m >= 0.2:
            raise SVIDWValidationError(
                "maximum_camber must lie in [0,0.2)"
            )
        if p <= 0.0 or p >= 1.0:
            raise SVIDWValidationError(
                "camber_location must lie in (0,1)"
            )
        if thickness >= 0.4:
            raise SVIDWValidationError(
                "thickness_ratio must lie in (0,0.4)"
            )
        if not isinstance(self.closed_trailing_edge, (bool, np.bool_)):
            raise SVIDWValidationError(
                "closed_trailing_edge must be boolean"
            )
        if not self.closed_trailing_edge:
            raise SVIDWValidationError(
                "N2.6e1 S0 requires a closed trailing edge; a finite-base "
                "closure has not been implemented"
            )
        object.__setattr__(self, "maximum_camber", m)
        object.__setattr__(self, "camber_location", p)
        object.__setattr__(self, "thickness_ratio", thickness)
        object.__setattr__(self, "chord", chord)


_FORBIDDEN_CONFIG_TOKENS = (
    "target",
    "measured",
    "force_multiplier",
    "lift_multiplier",
    "drag_multiplier",
    "thrust_multiplier",
    "force_offset",
    "lift_offset",
    "drag_offset",
    "thrust_offset",
)


@dataclass(frozen=True)
class SVIDWFoundationConfig:
    """Configuration for the inviscid N2.6e1 S0 equation oracle.

    ``from_mapping`` is the only mapping adapter.  It rejects unknown fields
    and target/force-calibration vocabulary before construction, so a caller
    cannot silently smuggle a free total-force knob into this foundation.
    """

    section: NACA4SectionConfig = NACA4SectionConfig()
    panels_per_side: int = 40
    freestream_speed: float = 1.0
    angle_of_attack_deg: float = 0.0
    density: float = 1.225

    def __post_init__(self) -> None:
        if not isinstance(self.section, NACA4SectionConfig):
            raise SVIDWValidationError(
                "section must be a NACA4SectionConfig"
            )
        panels = _integer_at_least(
            "panels_per_side", self.panels_per_side, 8
        )
        speed = _positive_scalar(
            "freestream_speed", self.freestream_speed
        )
        alpha = _finite_scalar(
            "angle_of_attack_deg", self.angle_of_attack_deg
        )
        density = _positive_scalar("density", self.density)
        if abs(alpha) >= 90.0:
            raise SVIDWValidationError(
                "angle_of_attack_deg must lie in (-90,90)"
            )
        object.__setattr__(self, "panels_per_side", panels)
        object.__setattr__(self, "freestream_speed", speed)
        object.__setattr__(self, "angle_of_attack_deg", alpha)
        object.__setattr__(self, "density", density)

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any]
    ) -> "SVIDWFoundationConfig":
        if not isinstance(payload, Mapping):
            raise SVIDWValidationError("configuration must be a mapping")
        keys = {str(key) for key in payload}
        lowered = {key.lower() for key in keys}
        forbidden = sorted(
            key
            for key in lowered
            if any(token in key for token in _FORBIDDEN_CONFIG_TOKENS)
        )
        if forbidden:
            raise SVIDWValidationError(
                "target-force calibration fields are forbidden in "
                f"N2.6e1 S0: {forbidden}"
            )
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(keys - allowed)
        if unknown:
            raise SVIDWValidationError(
                f"unknown N2.6e1 S0 configuration fields: {unknown}"
            )
        values = dict(payload)
        if "section" in values and isinstance(values["section"], Mapping):
            section_allowed = {
                item.name for item in fields(NACA4SectionConfig)
            }
            section_unknown = sorted(
                set(values["section"]) - section_allowed
            )
            if section_unknown:
                raise SVIDWValidationError(
                    "unknown NACA section fields: "
                    f"{section_unknown}"
                )
            values["section"] = NACA4SectionConfig(**values["section"])
        return cls(**values)


@dataclass(frozen=True)
class ActualSurface2D:
    """Closed clockwise polygon with explicit lower and upper wall traces."""

    section: NACA4SectionConfig
    upper_nodes: np.ndarray
    lower_nodes: np.ndarray
    contour_nodes: np.ndarray
    panel_midpoints: np.ndarray
    panel_tangents: np.ndarray
    panel_outward_normals: np.ndarray
    panel_lengths: np.ndarray
    panel_side: np.ndarray
    panel_chord_fraction: np.ndarray
    signed_area: float

    def __post_init__(self) -> None:
        if not isinstance(self.section, NACA4SectionConfig):
            raise SVIDWValidationError(
                "section must be a NACA4SectionConfig"
            )
        upper = _readonly_array(
            "upper_nodes", self.upper_nodes, ndim=2
        )
        lower = _readonly_array(
            "lower_nodes", self.lower_nodes, ndim=2
        )
        if (
            upper.shape != lower.shape
            or upper.shape[1] != 2
            or upper.shape[0] < 9
        ):
            raise SVIDWValidationError(
                "upper/lower nodes must share shape (n>=9,2)"
            )
        n_side = upper.shape[0] - 1
        contour = _readonly_array(
            "contour_nodes",
            self.contour_nodes,
            shape=(2 * n_side + 1, 2),
        )
        midpoints = _readonly_array(
            "panel_midpoints",
            self.panel_midpoints,
            shape=(2 * n_side, 2),
        )
        tangents = _readonly_array(
            "panel_tangents",
            self.panel_tangents,
            shape=(2 * n_side, 2),
        )
        normals = _readonly_array(
            "panel_outward_normals",
            self.panel_outward_normals,
            shape=(2 * n_side, 2),
        )
        lengths = _readonly_array(
            "panel_lengths",
            self.panel_lengths,
            shape=(2 * n_side,),
        )
        side = _readonly_array(
            "panel_side",
            self.panel_side,
            shape=(2 * n_side,),
            dtype="U5",
        )
        chord_fraction = _readonly_array(
            "panel_chord_fraction",
            self.panel_chord_fraction,
            shape=(2 * n_side,),
        )
        area = _finite_scalar("signed_area", self.signed_area)
        scale = self.section.chord
        closure_tolerance = 256.0 * np.finfo(float).eps * scale
        geometry_tolerance = 1024.0 * np.finfo(float).eps * max(
            scale,
            float(np.max(np.abs(contour), initial=0.0)),
        )
        if np.linalg.norm(contour[0] - contour[-1]) > closure_tolerance:
            raise SVIDWValidationError("contour is not closed")
        if area >= 0.0:
            raise SVIDWValidationError(
                "contour must be clockwise so the left normal is outward"
            )
        if np.any(lengths <= closure_tolerance):
            raise SVIDWValidationError("surface contains a zero-length panel")
        if np.max(
            np.abs(np.linalg.norm(tangents, axis=1) - 1.0),
            initial=0.0,
        ) > 1.0e-12:
            raise SVIDWValidationError("panel tangents are not unit vectors")
        if np.max(
            np.abs(np.linalg.norm(normals, axis=1) - 1.0),
            initial=0.0,
        ) > 1.0e-12:
            raise SVIDWValidationError("panel normals are not unit vectors")
        if np.max(
            np.abs(np.einsum("ij,ij->i", tangents, normals)),
            initial=0.0,
        ) > 1.0e-12:
            raise SVIDWValidationError(
                "panel tangent and outward normal are not orthogonal"
            )
        expected_side = np.array(
            ["lower"] * n_side + ["upper"] * n_side
        )
        if not np.array_equal(side, expected_side):
            raise SVIDWValidationError(
                "panel_side must identify lower then upper contour panels"
            )
        if np.any(
            (chord_fraction < -1.0e-12)
            | (chord_fraction > 1.0 + 1.0e-12)
        ):
            raise SVIDWValidationError(
                "panel chord fractions lie outside [0,1]"
            )
        if (
            np.linalg.norm(upper[0] - lower[0]) > geometry_tolerance
            or np.linalg.norm(upper[-1] - lower[-1])
            > geometry_tolerance
        ):
            raise SVIDWValidationError(
                "upper and lower traces must share leading- and "
                "trailing-edge endpoints"
            )
        expected_contour = np.vstack((lower[::-1], upper[1:]))
        if np.max(
            np.abs(contour - expected_contour), initial=0.0
        ) > geometry_tolerance:
            raise SVIDWValidationError(
                "contour_nodes must equal lower[::-1] followed by "
                "upper[1:]"
            )
        expected_segment = np.diff(expected_contour, axis=0)
        expected_lengths = np.linalg.norm(expected_segment, axis=1)
        if np.any(expected_lengths <= closure_tolerance):
            raise SVIDWValidationError(
                "upper/lower traces contain a zero-length panel"
            )
        expected_tangents = (
            expected_segment / expected_lengths[:, None]
        )
        expected_normals = np.column_stack((
            -expected_tangents[:, 1],
            expected_tangents[:, 0],
        ))
        expected_midpoints = 0.5 * (
            expected_contour[:-1] + expected_contour[1:]
        )
        expected_chord_fraction = np.clip(
            expected_midpoints[:, 0] / self.section.chord,
            0.0,
            1.0,
        )
        expected_area = 0.5 * float(np.sum(
            expected_contour[:-1, 0] * expected_contour[1:, 1]
            - expected_contour[1:, 0] * expected_contour[:-1, 1]
        ))
        direction_tolerance = 1024.0 * np.finfo(float).eps
        if np.max(
            np.abs(midpoints - expected_midpoints), initial=0.0
        ) > geometry_tolerance:
            raise SVIDWValidationError(
                "panel_midpoints are inconsistent with contour_nodes"
            )
        if np.max(
            np.abs(lengths - expected_lengths), initial=0.0
        ) > geometry_tolerance:
            raise SVIDWValidationError(
                "panel_lengths are inconsistent with contour_nodes"
            )
        if np.max(
            np.abs(tangents - expected_tangents), initial=0.0
        ) > direction_tolerance:
            raise SVIDWValidationError(
                "panel_tangents are inconsistent with contour_nodes"
            )
        if np.max(
            np.abs(normals - expected_normals), initial=0.0
        ) > direction_tolerance:
            raise SVIDWValidationError(
                "panel_outward_normals must be the clockwise contour's "
                "left normals"
            )
        if np.max(
            np.abs(chord_fraction - expected_chord_fraction), initial=0.0
        ) > direction_tolerance:
            raise SVIDWValidationError(
                "panel_chord_fraction is inconsistent with panel midpoints"
            )
        area_tolerance = (
            2048.0
            * np.finfo(float).eps
            * max(scale**2, abs(expected_area))
        )
        if abs(area - expected_area) > area_tolerance:
            raise SVIDWValidationError(
                "signed_area is inconsistent with contour_nodes"
            )
        object.__setattr__(self, "upper_nodes", upper)
        object.__setattr__(self, "lower_nodes", lower)
        object.__setattr__(self, "contour_nodes", contour)
        object.__setattr__(self, "panel_midpoints", midpoints)
        object.__setattr__(self, "panel_tangents", tangents)
        object.__setattr__(self, "panel_outward_normals", normals)
        object.__setattr__(self, "panel_lengths", lengths)
        object.__setattr__(self, "panel_side", side)
        object.__setattr__(self, "panel_chord_fraction", chord_fraction)
        object.__setattr__(self, "signed_area", area)

    @property
    def panels_per_side(self) -> int:
        return self.upper_nodes.shape[0] - 1

    @property
    def panel_count(self) -> int:
        return 2 * self.panels_per_side

    @property
    def perimeter(self) -> float:
        return float(np.sum(self.panel_lengths))

    @property
    def lower_panel_indices(self) -> np.ndarray:
        return np.arange(self.panels_per_side)

    @property
    def upper_panel_indices(self) -> np.ndarray:
        return np.arange(self.panels_per_side, self.panel_count)

    @property
    def canonical_ibl_map(self) -> "IBLContourMap2D":
        """Map canonical upper/lower IBL stations onto contour panels."""
        return IBLContourMap2D.from_surface(self)


def _naca4_mean_line(
    x: np.ndarray, *, maximum_camber: float, camber_location: float
) -> tuple[np.ndarray, np.ndarray]:
    m = maximum_camber
    p = camber_location
    yc = np.empty_like(x)
    slope = np.empty_like(x)
    before = x < p
    yc[before] = m / p**2 * (2.0 * p * x[before] - x[before] ** 2)
    slope[before] = 2.0 * m / p**2 * (p - x[before])
    yc[~before] = m / (1.0 - p) ** 2 * (
        1.0
        - 2.0 * p
        + 2.0 * p * x[~before]
        - x[~before] ** 2
    )
    slope[~before] = (
        2.0 * m / (1.0 - p) ** 2 * (p - x[~before])
    )
    return yc, slope


def build_naca4_actual_surface(
    section: NACA4SectionConfig,
    *,
    panels_per_side: int,
) -> ActualSurface2D:
    """Build a cosine-spaced, closed, clockwise actual NACA wall."""
    if not isinstance(section, NACA4SectionConfig):
        raise SVIDWValidationError(
            "section must be a NACA4SectionConfig"
        )
    n_side = _integer_at_least(
        "panels_per_side", panels_per_side, 8
    )
    beta = np.linspace(0.0, np.pi, n_side + 1)
    x = 0.5 * (1.0 - np.cos(beta))
    yc, slope = _naca4_mean_line(
        x,
        maximum_camber=section.maximum_camber,
        camber_location=section.camber_location,
    )
    trailing_coefficient = -0.1036
    yt = 5.0 * section.thickness_ratio * (
        0.2969 * np.sqrt(x)
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        + trailing_coefficient * x**4
    )
    theta = np.arctan(slope)
    upper = np.column_stack((
        x - yt * np.sin(theta),
        yc + yt * np.cos(theta),
    ))
    lower = np.column_stack((
        x + yt * np.sin(theta),
        yc - yt * np.cos(theta),
    ))
    upper *= section.chord
    lower *= section.chord
    # The polynomial's closed-TE cancellation is only floating-point exact.
    # Set the geometric endpoints explicitly so topology does not depend on
    # cancellation order.
    upper[0] = lower[0] = (0.0, 0.0)
    upper[-1] = lower[-1] = (section.chord, 0.0)

    contour = np.vstack((lower[::-1], upper[1:]))
    segment = np.diff(contour, axis=0)
    lengths = np.linalg.norm(segment, axis=1)
    tangents = segment / lengths[:, None]
    normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))
    midpoints = 0.5 * (contour[:-1] + contour[1:])
    side = np.array(
        ["lower"] * n_side + ["upper"] * n_side
    )
    chord_fraction = np.clip(
        midpoints[:, 0] / section.chord, 0.0, 1.0
    )
    signed_area = 0.5 * float(np.sum(
        contour[:-1, 0] * contour[1:, 1]
        - contour[1:, 0] * contour[:-1, 1]
    ))
    return ActualSurface2D(
        section=section,
        upper_nodes=upper,
        lower_nodes=lower,
        contour_nodes=contour,
        panel_midpoints=midpoints,
        panel_tangents=tangents,
        panel_outward_normals=normals,
        panel_lengths=lengths,
        panel_side=side,
        panel_chord_fraction=chord_fraction,
        signed_area=signed_area,
    )


@dataclass(frozen=True)
class IBLContourMap2D:
    """Explicit map between IBL and clockwise-contour station conventions.

    IBL arrays have shape ``(2,n)`` and side order ``upper, lower``.  Each
    side is ordered from its resolved stagnation cut towards the trailing
    edge.  The S0 NACA wall places that cut at the common leading-edge node.
    The clockwise contour instead stores lower panels from TE to LE, followed
    by upper panels from LE to TE.  Tangential quantities therefore change
    sign on the lower side; unoriented scalars such as outward transpiration
    do not.
    """

    ibl_to_contour_index: np.ndarray
    ibl_tangent_to_contour_sign: np.ndarray

    def __post_init__(self) -> None:
        raw_index = np.asarray(self.ibl_to_contour_index)
        index = _readonly_array(
            "ibl_to_contour_index",
            raw_index,
            ndim=2,
            dtype=int,
        )
        try:
            integral_index = np.all(
                np.asarray(raw_index, dtype=float) == index
            )
        except (TypeError, ValueError):
            integral_index = False
        if not integral_index:
            raise SVIDWValidationError(
                "IBL-to-contour indices must be integers"
            )
        if index.shape[0] != 2 or index.shape[1] < 8:
            raise SVIDWValidationError(
                "IBL contour map must have shape (2,n>=8)"
            )
        sign = _readonly_array(
            "ibl_tangent_to_contour_sign",
            self.ibl_tangent_to_contour_sign,
            shape=index.shape,
        )
        if not np.all(np.isin(sign, (-1.0, 1.0))):
            raise SVIDWValidationError(
                "IBL-to-contour tangent signs must be +/-1"
            )
        flat = index.ravel()
        if (
            np.min(flat, initial=0) < 0
            or np.max(flat, initial=-1) >= flat.size
            or np.unique(flat).size != flat.size
        ):
            raise SVIDWValidationError(
                "IBL-to-contour indices must be a permutation of panels"
            )
        object.__setattr__(self, "ibl_to_contour_index", index)
        object.__setattr__(self, "ibl_tangent_to_contour_sign", sign)

    @classmethod
    def from_surface(
        cls, surface: ActualSurface2D
    ) -> "IBLContourMap2D":
        if not isinstance(surface, ActualSurface2D):
            raise SVIDWValidationError(
                "surface must be an ActualSurface2D"
            )
        n_side = surface.panels_per_side
        upper = np.arange(n_side, 2 * n_side, dtype=int)
        lower = np.arange(n_side - 1, -1, -1, dtype=int)
        indices = np.vstack((upper, lower))
        signs = np.vstack((
            np.ones(n_side, dtype=float),
            -np.ones(n_side, dtype=float),
        ))
        return cls(
            ibl_to_contour_index=indices,
            ibl_tangent_to_contour_sign=signs,
        )

    @property
    def panels_per_side(self) -> int:
        return self.ibl_to_contour_index.shape[1]

    @property
    def panel_count(self) -> int:
        return 2 * self.panels_per_side

    def scalar_to_contour(self, values: Any) -> np.ndarray:
        """Map an unoriented upper/lower scalar to contour panel order."""
        array = np.asarray(values, dtype=float)
        expected = (2, self.panels_per_side)
        if array.shape != expected or not np.all(np.isfinite(array)):
            raise SVIDWValidationError(
                f"IBL scalar field must be finite with shape {expected}"
            )
        contour = np.empty(self.panel_count, dtype=float)
        contour[self.ibl_to_contour_index] = array
        contour.setflags(write=False)
        return contour

    def tangential_to_contour(self, values: Any) -> np.ndarray:
        """Map a stagnation-to-TE signed scalar to contour-tangent sign."""
        array = np.asarray(values, dtype=float)
        expected = (2, self.panels_per_side)
        if array.shape != expected or not np.all(np.isfinite(array)):
            raise SVIDWValidationError(
                f"IBL tangential field must be finite with shape {expected}"
            )
        return self.scalar_to_contour(
            array * self.ibl_tangent_to_contour_sign
        )

    def scalar_from_contour(self, values: Any) -> np.ndarray:
        """Recover canonical upper/lower values from contour panel order."""
        contour = np.asarray(values, dtype=float)
        if (
            contour.shape != (self.panel_count,)
            or not np.all(np.isfinite(contour))
        ):
            raise SVIDWValidationError(
                "contour scalar field must be finite with one value "
                "per panel"
            )
        result = contour[self.ibl_to_contour_index].copy()
        result.setflags(write=False)
        return result

    def tangential_from_contour(self, values: Any) -> np.ndarray:
        """Recover stagnation-to-TE tangential signs from the contour."""
        result = (
            self.scalar_from_contour(values)
            * self.ibl_tangent_to_contour_sign
        )
        result.setflags(write=False)
        return result


@dataclass(frozen=True)
class DualSideIBLState:
    """Typed IBL inventory in canonical upper/lower stagnation-to-TE order.

    ``transition_amplification`` is the laminar third degree of freedom
    (the e^N amplification factor ``n``); ``maximum_shear_stress_coefficient``
    is its turbulent counterpart ``C_tau``.  ``regime`` selects exactly one
    of them at every active station.  This state declares no closure equation.
    """

    displacement_thickness: np.ndarray
    momentum_thickness: np.ndarray
    kinetic_energy_thickness: np.ndarray
    edge_velocity: np.ndarray
    skin_friction_coefficient: np.ndarray
    transpiration_velocity: np.ndarray
    regime: np.ndarray | IBLRegime | str | None = None
    transition_amplification: np.ndarray | None = None
    maximum_shear_stress_coefficient: np.ndarray | None = None

    def __post_init__(self) -> None:
        delta = _readonly_array(
            "displacement_thickness",
            self.displacement_thickness,
            ndim=2,
        )
        if delta.shape[0] != 2 or delta.shape[1] < 8:
            raise SVIDWValidationError(
                "IBL arrays must have shape (2,n>=8), ordered upper/lower"
            )
        shape = delta.shape
        theta = _readonly_array(
            "momentum_thickness",
            self.momentum_thickness,
            shape=shape,
        )
        theta_k = _readonly_array(
            "kinetic_energy_thickness",
            self.kinetic_energy_thickness,
            shape=shape,
        )
        edge = _readonly_array(
            "edge_velocity", self.edge_velocity, shape=shape
        )
        cf = _readonly_array(
            "skin_friction_coefficient",
            self.skin_friction_coefficient,
            shape=shape,
        )
        transpiration = _readonly_array(
            "transpiration_velocity",
            self.transpiration_velocity,
            shape=shape,
        )
        if self.regime is None:
            regime = np.full(
                shape, IBLRegime.INACTIVE.value, dtype="U10"
            )
            regime.setflags(write=False)
        elif isinstance(self.regime, (IBLRegime, str)):
            value = (
                self.regime.value
                if isinstance(self.regime, IBLRegime)
                else self.regime
            )
            regime = np.full(shape, value, dtype="U10")
            regime.setflags(write=False)
        else:
            raw_regime = np.asarray(self.regime, dtype=object)
            if raw_regime.shape != shape:
                raise SVIDWValidationError(
                    f"regime must have shape {shape}, got "
                    f"{raw_regime.shape}"
                )
            regime = np.array(
                [
                    item.value
                    if isinstance(item, IBLRegime)
                    else str(item)
                    for item in raw_regime.ravel()
                ],
                dtype="U10",
            ).reshape(shape)
            regime.setflags(write=False)
        allowed_regimes = {item.value for item in IBLRegime}
        unknown_regimes = sorted(
            set(np.unique(regime)) - allowed_regimes
        )
        if unknown_regimes:
            raise SVIDWValidationError(
                f"unknown IBL regimes: {unknown_regimes}"
            )
        if self.transition_amplification is None:
            amplification = np.zeros(shape, dtype=float)
            amplification.setflags(write=False)
        else:
            amplification = _readonly_array(
                "transition_amplification",
                self.transition_amplification,
                shape=shape,
            )
        if self.maximum_shear_stress_coefficient is None:
            c_tau = np.zeros(shape, dtype=float)
            c_tau.setflags(write=False)
        else:
            c_tau = _readonly_array(
                "maximum_shear_stress_coefficient",
                self.maximum_shear_stress_coefficient,
                shape=shape,
            )
        if np.any(delta < 0.0) or np.any(theta < 0.0) or np.any(
            theta_k < 0.0
        ):
            raise SVIDWValidationError(
                "IBL thickness inventories must be non-negative"
            )
        active = theta > 0.0
        if np.any(delta[active] < theta[active]):
            raise SVIDWValidationError(
                "active displacement thickness must not be below "
                "momentum thickness"
            )
        if np.any(theta_k[active] < theta[active]):
            raise SVIDWValidationError(
                "active kinetic-energy thickness must not be below "
                "momentum thickness"
            )
        if np.any(amplification < 0.0) or np.any(c_tau < 0.0):
            raise SVIDWValidationError(
                "IBL n and C_tau states must be non-negative"
            )
        laminar = regime == IBLRegime.LAMINAR.value
        turbulent = regime == IBLRegime.TURBULENT.value
        inactive = regime == IBLRegime.INACTIVE.value
        if np.any(c_tau[laminar] != 0.0):
            raise SVIDWValidationError(
                "laminar stations cannot carry turbulent C_tau state"
            )
        if np.any(amplification[turbulent] != 0.0):
            raise SVIDWValidationError(
                "turbulent stations cannot carry laminar amplification n"
            )
        coupling_inventory = (
            (delta != 0.0)
            | (theta != 0.0)
            | (theta_k != 0.0)
            | (cf != 0.0)
            | (transpiration != 0.0)
            | (amplification != 0.0)
            | (c_tau != 0.0)
        )
        if np.any(coupling_inventory & inactive):
            raise SVIDWValidationError(
                "inactive IBL stations cannot carry viscous-coupling state"
            )
        object.__setattr__(self, "displacement_thickness", delta)
        object.__setattr__(self, "momentum_thickness", theta)
        object.__setattr__(self, "kinetic_energy_thickness", theta_k)
        object.__setattr__(self, "edge_velocity", edge)
        object.__setattr__(self, "skin_friction_coefficient", cf)
        object.__setattr__(self, "transpiration_velocity", transpiration)
        object.__setattr__(self, "regime", regime)
        object.__setattr__(self, "transition_amplification", amplification)
        object.__setattr__(
            self, "maximum_shear_stress_coefficient", c_tau
        )

    @classmethod
    def quiescent(cls, panels_per_side: int) -> "DualSideIBLState":
        n_side = _integer_at_least(
            "panels_per_side", panels_per_side, 8
        )
        zero = np.zeros((2, n_side), dtype=float)
        return cls(
            displacement_thickness=zero,
            momentum_thickness=zero,
            kinetic_energy_thickness=zero,
            edge_velocity=zero,
            skin_friction_coefficient=zero,
            transpiration_velocity=zero,
            regime=np.full(
                (2, n_side), IBLRegime.INACTIVE.value, dtype="U10"
            ),
            transition_amplification=zero,
            maximum_shear_stress_coefficient=zero,
        )

    @property
    def has_viscous_coupling(self) -> bool:
        """Whether this state changes transpiration, shear, or IBL memory.

        Edge velocity is an inviscid observable and intentionally does not
        activate viscous coupling.
        """
        arrays = (
            self.displacement_thickness,
            self.momentum_thickness,
            self.kinetic_energy_thickness,
            self.skin_friction_coefficient,
            self.transpiration_velocity,
            self.transition_amplification,
            self.maximum_shear_stress_coefficient,
        )
        return (
            any(np.any(array) for array in arrays)
            or np.any(self.regime != IBLRegime.INACTIVE.value)
        )

    @property
    def is_quiescent(self) -> bool:
        """Compatibility alias for ``not has_viscous_coupling``."""
        return not self.has_viscous_coupling

    @property
    def shape_factor(self) -> np.ndarray:
        result = np.full(
            self.momentum_thickness.shape, np.nan, dtype=float
        )
        active = self.momentum_thickness > 0.0
        result[active] = (
            self.displacement_thickness[active]
            / self.momentum_thickness[active]
        )
        result.setflags(write=False)
        return result

    @property
    def kinetic_energy_shape_factor(self) -> np.ndarray:
        result = np.full(
            self.momentum_thickness.shape, np.nan, dtype=float
        )
        active = self.momentum_thickness > 0.0
        result[active] = (
            self.kinetic_energy_thickness[active]
            / self.momentum_thickness[active]
        )
        result.setflags(write=False)
        return result

    def contour_edge_velocity(
        self, surface: ActualSurface2D
    ) -> np.ndarray:
        return surface.canonical_ibl_map.tangential_to_contour(
            self.edge_velocity
        )

    def contour_transpiration_velocity(
        self, surface: ActualSurface2D
    ) -> np.ndarray:
        return surface.canonical_ibl_map.scalar_to_contour(
            self.transpiration_velocity
        )


@dataclass(frozen=True)
class WakeBranch2D:
    """One material wake branch; evolution and induction are not S0 physics.

    ``segment_circulation`` stores circulation integrated over each segment
    in length^2/time.  Positive values use the standard counter-clockwise
    point-vortex convention.  It is not a vortex-sheet strength in
    length/time.
    """

    role: str
    nodes: np.ndarray
    segment_circulation: np.ndarray

    def __post_init__(self) -> None:
        if self.role not in {"trailing_edge", "separation"}:
            raise SVIDWValidationError(
                "wake role must be 'trailing_edge' or 'separation'"
            )
        nodes = _readonly_array("wake nodes", self.nodes, ndim=2)
        if nodes.shape[1:] != (2,):
            raise SVIDWValidationError(
                "wake nodes must have shape (n,2)"
            )
        circulation = _readonly_array(
            "segment_circulation",
            self.segment_circulation,
            ndim=1,
        )
        expected = max(nodes.shape[0] - 1, 0)
        if circulation.shape != (expected,):
            raise SVIDWValidationError(
                "segment_circulation must have one value per wake segment"
            )
        if nodes.shape[0] == 1:
            raise SVIDWValidationError(
                "a wake branch must be empty or contain at least two nodes"
            )
        if nodes.shape[0] >= 2:
            segment_length = np.linalg.norm(
                np.diff(nodes, axis=0), axis=1
            )
            coordinate_scale = max(
                float(np.max(np.abs(nodes), initial=0.0)), 1.0
            )
            tolerance = (
                256.0 * np.finfo(float).eps * coordinate_scale
            )
            if np.any(segment_length <= tolerance):
                raise SVIDWValidationError(
                    "wake branch contains a zero-length segment"
                )
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "segment_circulation", circulation)

    @classmethod
    def empty(cls, role: str) -> "WakeBranch2D":
        return cls(
            role=role,
            nodes=np.empty((0, 2), dtype=float),
            segment_circulation=np.empty(0, dtype=float),
        )

    @property
    def is_quiescent(self) -> bool:
        """Compatibility alias for ``not has_induction``."""
        return not self.has_induction

    @property
    def has_induction(self) -> bool:
        return bool(np.any(self.segment_circulation != 0.0))

    @property
    def total_circulation(self) -> float:
        return float(np.sum(self.segment_circulation))


@dataclass(frozen=True)
class DoubleWakeState2D:
    """Typed TE-wake plus separated-shear-layer-wake state."""

    trailing_edge: WakeBranch2D
    separation: WakeBranch2D

    def __post_init__(self) -> None:
        if (
            not isinstance(self.trailing_edge, WakeBranch2D)
            or self.trailing_edge.role != "trailing_edge"
        ):
            raise SVIDWValidationError(
                "trailing_edge must be a trailing-edge WakeBranch2D"
            )
        if (
            not isinstance(self.separation, WakeBranch2D)
            or self.separation.role != "separation"
        ):
            raise SVIDWValidationError(
                "separation must be a separation WakeBranch2D"
            )

    @classmethod
    def quiescent(cls) -> "DoubleWakeState2D":
        return cls(
            trailing_edge=WakeBranch2D.empty("trailing_edge"),
            separation=WakeBranch2D.empty("separation"),
        )

    @property
    def is_quiescent(self) -> bool:
        """Compatibility alias for ``not has_induction``."""
        return not self.has_induction

    @property
    def has_induction(self) -> bool:
        return (
            self.trailing_edge.has_induction
            or self.separation.has_induction
        )


@dataclass(frozen=True)
class SurfaceTractionState2D:
    """Panel traction, force, and moment derived from pressure and shear.

    Stored ``wall_shear`` is signed along the clockwise contour tangent.
    Use ``from_pressure_and_ibl_shear`` when shear is instead positive in
    each side's canonical stagnation-to-TE direction.
    """

    surface: ActualSurface2D
    pressure: np.ndarray
    wall_shear: np.ndarray
    traction: np.ndarray
    panel_force: np.ndarray
    moment_reference: Any = (0.0, 0.0)
    panel_moment: Any | None = None

    @classmethod
    def from_pressure_and_shear(
        cls,
        surface: ActualSurface2D,
        *,
        pressure: Any,
        wall_shear: Any,
        reference_point: Any = (0.0, 0.0),
    ) -> "SurfaceTractionState2D":
        if not isinstance(surface, ActualSurface2D):
            raise SVIDWValidationError(
                "surface must be an ActualSurface2D"
            )
        shape = (surface.panel_count,)
        pressure_array = _readonly_array(
            "pressure", pressure, shape=shape
        )
        shear_array = _readonly_array(
            "wall_shear", wall_shear, shape=shape
        )
        traction = (
            -pressure_array[:, None] * surface.panel_outward_normals
            + shear_array[:, None] * surface.panel_tangents
        )
        force = traction * surface.panel_lengths[:, None]
        reference = _readonly_array(
            "reference_point", reference_point, shape=(2,)
        )
        arm = surface.panel_midpoints - reference
        moment = (
            arm[:, 0] * force[:, 1]
            - arm[:, 1] * force[:, 0]
        )
        return cls(
            surface=surface,
            pressure=pressure_array,
            wall_shear=shear_array,
            traction=traction,
            panel_force=force,
            moment_reference=reference,
            panel_moment=moment,
        )

    @classmethod
    def from_pressure_and_ibl_shear(
        cls,
        surface: ActualSurface2D,
        *,
        pressure: Any,
        ibl_wall_shear: Any,
        reference_point: Any = (0.0, 0.0),
    ) -> "SurfaceTractionState2D":
        """Build traction from canonical upper/lower BL-directed shear."""
        contour_shear = (
            surface.canonical_ibl_map.tangential_to_contour(
                ibl_wall_shear
            )
        )
        return cls.from_pressure_and_shear(
            surface,
            pressure=pressure,
            wall_shear=contour_shear,
            reference_point=reference_point,
        )

    def __post_init__(self) -> None:
        if not isinstance(self.surface, ActualSurface2D):
            raise SVIDWValidationError(
                "surface must be an ActualSurface2D"
            )
        pressure = _readonly_array("pressure", self.pressure, ndim=1)
        n_panel = pressure.shape[0]
        if n_panel != self.surface.panel_count:
            raise SVIDWValidationError(
                "traction arrays must have one value per surface panel"
            )
        shear = _readonly_array(
            "wall_shear", self.wall_shear, shape=(n_panel,)
        )
        traction = _readonly_array(
            "traction", self.traction, shape=(n_panel, 2)
        )
        force = _readonly_array(
            "panel_force", self.panel_force, shape=(n_panel, 2)
        )
        reference = _readonly_array(
            "moment_reference", self.moment_reference, shape=(2,)
        )
        expected_traction = (
            -pressure[:, None] * self.surface.panel_outward_normals
            + shear[:, None] * self.surface.panel_tangents
        )
        expected_force = (
            expected_traction * self.surface.panel_lengths[:, None]
        )
        arm = self.surface.panel_midpoints - reference
        expected_moment = (
            arm[:, 0] * expected_force[:, 1]
            - arm[:, 1] * expected_force[:, 0]
        )
        if self.panel_moment is None:
            moment = expected_moment.copy()
            moment.setflags(write=False)
        else:
            moment = _readonly_array(
                "panel_moment",
                self.panel_moment,
                shape=(n_panel,),
            )
        scale = max(
            float(np.max(np.abs(expected_force), initial=0.0)),
            float(np.max(np.abs(expected_moment), initial=0.0)),
            1.0,
        )
        tolerance = 64.0 * np.finfo(float).eps * scale
        if (
            np.max(
                np.abs(traction - expected_traction), initial=0.0
            )
            > tolerance
            or np.max(np.abs(force - expected_force), initial=0.0)
            > tolerance
            or np.max(
                np.abs(moment - expected_moment), initial=0.0
            )
            > tolerance
        ):
            raise SVIDWValidationError(
                "traction, panel_force, and panel_moment must be derived "
                "from the declared pressure, shear, reference, and wall"
            )
        object.__setattr__(self, "pressure", pressure)
        object.__setattr__(self, "wall_shear", shear)
        object.__setattr__(self, "traction", traction)
        object.__setattr__(self, "panel_force", force)
        object.__setattr__(self, "moment_reference", reference)
        object.__setattr__(self, "panel_moment", moment)

    @property
    def resultant_force(self) -> np.ndarray:
        return np.sum(self.panel_force, axis=0)

    @property
    def resultant_moment(self) -> float:
        return float(np.sum(self.panel_moment))
