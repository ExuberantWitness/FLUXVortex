"""Canonical airfoil mean-camber geometries used by research guards.

The SD7003 coordinates are from the UIUC Airfoil Data Site:
https://m-selig.ae.illinois.edu/ads/coord_seligFmt/sd7003.dat

Production RoboEagle geometry remains in ``_v2_robogeom.py``.  This module
exists so Hirato Case 1/2 cannot silently use a flat plate while being labelled
SD7003.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np


@lru_cache(maxsize=1)
def sd7003_coordinates() -> np.ndarray:
    path = (
        Path(__file__).resolve().parents[1]
        / "researchpaper"
        / "uiuc_airfoils"
        / "sd7003.dat"
    )
    coordinates = np.loadtxt(path, skiprows=1)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError(f"invalid SD7003 coordinate file: {coordinates.shape}")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("SD7003 coordinates contain non-finite values")
    return coordinates


@lru_cache(maxsize=1)
def sd7003_chord_coordinates() -> np.ndarray:
    """Return SD7003 coordinates in its geometric LE--TE chord frame.

    The published Selig-format coordinates retain the finite rounded nose:
    their minimum-x point is not exactly ``(0, 0)``.  UVLM lattices, however,
    are defined in a chord frame whose leading edge is the origin and whose
    trailing edge is ``(1, 0)``.  This rigid transform plus chord scaling
    preserves the airfoil shape without inventing a fitted camber line.
    """
    coordinates = sd7003_coordinates()
    i_le = int(np.argmin(coordinates[:, 0]))
    leading_edge = coordinates[i_le]
    trailing_edge = 0.5 * (coordinates[0] + coordinates[-1])
    chord_vector = trailing_edge - leading_edge
    chord_length = float(np.linalg.norm(chord_vector))
    if not np.isfinite(chord_length) or chord_length <= 0.0:
        raise ValueError("invalid SD7003 geometric chord")
    chord_axis = chord_vector / chord_length
    normal_axis = np.array([-chord_axis[1], chord_axis[0]])
    relative = coordinates - leading_edge
    transformed = np.column_stack((
        relative @ chord_axis / chord_length,
        relative @ normal_axis / chord_length,
    ))
    # Eliminate roundoff at the two defining endpoints.  This is a coordinate
    # identity, not a fitted aerodynamic correction.
    transformed[i_le] = (0.0, 0.0)
    transformed[0, 0] = 1.0
    transformed[-1, 0] = 1.0
    return transformed


def sd7003_mean_camber(x_over_c) -> np.ndarray:
    """Interpolate the SD7003 mean-camber ordinate at normalized chord points."""
    x = np.asarray(x_over_c, dtype=float)
    if np.any(~np.isfinite(x)) or np.any((x < 0.0) | (x > 1.0)):
        raise ValueError("x_over_c must be finite and lie in [0, 1]")

    coordinates = sd7003_chord_coordinates()
    # Selig format traverses upper TE->LE and lower LE->TE.  The minimum-x
    # coordinate belongs to both sides; including it in each interpolation
    # preserves the finite rounded leading edge without fitting a surrogate.
    i_le = int(np.argmin(coordinates[:, 0]))
    upper = coordinates[:i_le + 1][::-1]
    lower = coordinates[i_le:]
    z_upper = np.interp(x, upper[:, 0], upper[:, 1])
    z_lower = np.interp(x, lower[:, 0], lower[:, 1])
    return 0.5 * (z_upper + z_lower)


def sd7003_mean_camber_wing(
    nc: int,
    ns: int,
    chord: float,
    half_span: float,
) -> np.ndarray:
    """Return a rectangular half-wing lattice on the SD7003 mean-camber line."""
    if nc <= 0 or ns <= 0 or chord <= 0.0 or half_span <= 0.0:
        raise ValueError("nc, ns, chord and half_span must be positive")
    x_over_c = np.linspace(0.0, 1.0, nc + 1)
    y = np.linspace(0.0, half_span, ns + 1)
    x = chord * x_over_c
    z = chord * sd7003_mean_camber(x_over_c)
    lattice = np.zeros((nc + 1, ns + 1, 3), dtype=float)
    lattice[..., 0] = x[:, None]
    lattice[..., 1] = y[None, :]
    lattice[..., 2] = z[:, None]
    return lattice
