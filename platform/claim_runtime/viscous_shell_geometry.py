"""Dual-side NACA-2406 shell sharing N1 material coordinates.

N1 remains a zero-thickness lifting lattice on the mean camber surface.
N2.6 requires a separate physical wall on which upper/lower pressure
gradients, boundary-layer inventory and separation manifolds can live.  This
module builds that wall from the defining NACA four-digit geometry without
changing the N1 lattice, circulation or force.

The material coordinates are ``(xi, eta)``: chord fraction and wing-local
span.  At each material coordinate the upper and lower points are offset
normal to the two-dimensional mean line, so their arithmetic mean is exactly
the N1 camber point.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class ViscousShellGeometryError(ValueError):
    """Invalid airfoil or planform geometry."""


def _finite(name: str, value, *, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if ndim is not None and array.ndim != ndim:
        raise ViscousShellGeometryError(
            f"{name} must have ndim={ndim}, got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ViscousShellGeometryError(f"{name} contains non-finite values")
    return array


@dataclass(frozen=True)
class DualSurfaceShell:
    chord_fraction: np.ndarray
    span: np.ndarray
    chord: np.ndarray
    mean_surface: np.ndarray
    upper_surface: np.ndarray
    lower_surface: np.ndarray
    half_thickness: np.ndarray
    section_director: np.ndarray


def naca4_mean_line(
    chord_fraction,
    *,
    maximum_camber: float,
    camber_location: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``yc/c`` and ``d(yc/c)/d(x/c)`` for a NACA four-digit mean line."""
    x = _finite("chord_fraction", chord_fraction)
    if np.any((x < 0.0) | (x > 1.0)):
        raise ViscousShellGeometryError(
            "chord_fraction must lie in [0,1]"
        )
    if (
        not np.isfinite(maximum_camber)
        or maximum_camber < 0.0
        or not np.isfinite(camber_location)
        or camber_location <= 0.0
        or camber_location >= 1.0
    ):
        raise ViscousShellGeometryError(
            "invalid NACA four-digit camber parameters"
        )
    m = float(maximum_camber)
    p = float(camber_location)
    before = x < p
    yc = np.empty_like(x)
    slope = np.empty_like(x)
    yc[before] = m/(p*p)*(2.0*p*x[before]-x[before]**2)
    slope[before] = 2.0*m/(p*p)*(p-x[before])
    yc[~before] = m/(1.0-p)**2*(
        (1.0-2.0*p)+2.0*p*x[~before]-x[~before]**2
    )
    slope[~before] = 2.0*m/(1.0-p)**2*(p-x[~before])
    return yc, slope


def naca4_half_thickness(
    chord_fraction,
    *,
    thickness_ratio: float,
    closed_trailing_edge: bool = False,
) -> np.ndarray:
    """Return the NACA four-digit half-thickness ``yt/c``."""
    x = _finite("chord_fraction", chord_fraction)
    if np.any((x < 0.0) | (x > 1.0)):
        raise ViscousShellGeometryError(
            "chord_fraction must lie in [0,1]"
        )
    if (
        not np.isfinite(thickness_ratio)
        or thickness_ratio <= 0.0
        or thickness_ratio >= 1.0
    ):
        raise ViscousShellGeometryError(
            "thickness_ratio must lie in (0,1)"
        )
    trailing = -0.1036 if closed_trailing_edge else -0.1015
    return 5.0*float(thickness_ratio)*(
        0.2969*np.sqrt(x)
        - 0.1260*x
        - 0.3516*x**2
        + 0.2843*x**3
        + trailing*x**4
    )


def naca4_dual_surface_shell(
    chord_fraction,
    span,
    chord,
    *,
    maximum_camber: float = 0.02,
    camber_location: float = 0.40,
    thickness_ratio: float = 0.06,
    closed_trailing_edge: bool = False,
) -> DualSurfaceShell:
    """Build an untwisted finite-wing dual wall in shared material coordinates."""
    xi = _finite("chord_fraction", chord_fraction, ndim=1)
    eta = _finite("span", span, ndim=1)
    local_chord = _finite("chord", chord, ndim=1)
    if len(eta) != len(local_chord):
        raise ViscousShellGeometryError(
            "span and chord must have the same length"
        )
    if len(xi) < 2 or len(eta) < 2:
        raise ViscousShellGeometryError(
            "at least two chord and span nodes are required"
        )
    if np.any(np.diff(xi) <= 0.0) or np.any(np.diff(eta) <= 0.0):
        raise ViscousShellGeometryError(
            "material coordinates must be strictly increasing"
        )
    if np.any(local_chord <= 0.0):
        raise ViscousShellGeometryError("chord must be positive")

    yc, slope = naca4_mean_line(
        xi,
        maximum_camber=maximum_camber,
        camber_location=camber_location,
    )
    yt = naca4_half_thickness(
        xi,
        thickness_ratio=thickness_ratio,
        closed_trailing_edge=closed_trailing_edge,
    )
    theta = np.arctan(slope)
    director_2d = np.column_stack(
        (-np.sin(theta), np.cos(theta))
    )
    mean = np.zeros((len(xi), len(eta), 3), dtype=float)
    director = np.zeros_like(mean)
    half_thickness = np.empty((len(xi), len(eta)), dtype=float)
    for span_index, section_chord in enumerate(local_chord):
        mean[:, span_index, 0] = xi*section_chord
        mean[:, span_index, 1] = eta[span_index]
        mean[:, span_index, 2] = yc*section_chord
        director[:, span_index, 0] = director_2d[:, 0]
        director[:, span_index, 2] = director_2d[:, 1]
        half_thickness[:, span_index] = yt*section_chord
    upper = mean+half_thickness[..., None]*director
    lower = mean-half_thickness[..., None]*director
    return DualSurfaceShell(
        chord_fraction=xi.copy(),
        span=eta.copy(),
        chord=local_chord.copy(),
        mean_surface=mean,
        upper_surface=upper,
        lower_surface=lower,
        half_thickness=half_thickness,
        section_director=director,
    )


def rigidly_transform_shell(
    shell: DualSurfaceShell,
    *,
    rotation,
    translation,
) -> DualSurfaceShell:
    """Apply one proper rigid transform to all three material surfaces."""
    if not isinstance(shell, DualSurfaceShell):
        raise ViscousShellGeometryError(
            "shell must be a DualSurfaceShell"
        )
    matrix = _finite("rotation", rotation)
    shift = _finite("translation", translation)
    if matrix.shape != (3, 3) or shift.shape != (3,):
        raise ViscousShellGeometryError(
            "rotation and translation must have shapes (3,3) and (3,)"
        )
    if (
        np.max(np.abs(matrix.T@matrix-np.eye(3)), initial=0.0) > 1.0e-12
        or abs(float(np.linalg.det(matrix))-1.0) > 1.0e-12
    ):
        raise ViscousShellGeometryError(
            "rotation must be a proper orthogonal matrix"
        )

    def transform(points):
        return np.einsum("ij,...j->...i", matrix, points)+shift

    return DualSurfaceShell(
        chord_fraction=shell.chord_fraction.copy(),
        span=shell.span.copy(),
        chord=shell.chord.copy(),
        mean_surface=transform(shell.mean_surface),
        upper_surface=transform(shell.upper_surface),
        lower_surface=transform(shell.lower_surface),
        half_thickness=shell.half_thickness.copy(),
        section_director=np.einsum(
            "ij,...j->...i", matrix, shell.section_director
        ),
    )
