"""Explicit sign/orientation adapter between the frozen N1 lattice and DDEs.

N1 stores a positive panel circulation on the ring ordered
``front-left -> front-right -> aft-right -> aft-left``.  For the N1 panel
normal this is the *negative* oriented boundary.  A positive constant DDE
strength, by contrast, induces the positive boundary vortex of its oriented
surface.  Consequently, for aligned N1 and DDE normals,

``mu_DDE = -Gamma_N1``.

The minus sign is not a fitted convention.  It follows from the two explicit
boundary orientations and is checked against the induced-velocity operators
in the formula-audit guard.  Reversing the DDE normal reverses the sign once
more.  This module performs no solve, shedding, pressure, or force operation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class N1DDEInterfaceError(ValueError):
    """Invalid or ambiguous N1/DDE orientation interface."""


def _finite(name: str, value, *, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if ndim is not None and array.ndim != ndim:
        raise N1DDEInterfaceError(
            f"{name} must have ndim={ndim}, got shape {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise N1DDEInterfaceError(f"{name} contains non-finite values")
    return array


@dataclass(frozen=True)
class N1DDEOrientation:
    """Discrete orientation map for one or more interface degrees of freedom."""

    alignment: np.ndarray
    max_parallel_residual: float


def interface_orientation(
    n1_normal,
    dde_normal,
    *,
    parallel_tolerance: float = 1.0e-12,
) -> N1DDEOrientation:
    """Return ``+1`` for aligned normals and ``-1`` for opposed normals.

    Oblique interfaces are rejected.  Silently projecting between oblique
    normals would mix a geometry error into the circulation sign convention.
    """
    n1 = _finite("n1_normal", n1_normal, ndim=2)
    dde = _finite("dde_normal", dde_normal, ndim=2)
    if n1.shape != dde.shape or n1.shape[1] != 3:
        raise N1DDEInterfaceError(
            "n1_normal and dde_normal must have matching shape (n,3)"
        )
    if (
        not np.isfinite(parallel_tolerance)
        or parallel_tolerance < 0.0
    ):
        raise N1DDEInterfaceError(
            "parallel_tolerance must be finite and non-negative"
        )
    n1_norm = np.linalg.norm(n1, axis=1)
    dde_norm = np.linalg.norm(dde, axis=1)
    if np.any(n1_norm <= 0.0) or np.any(dde_norm <= 0.0):
        raise N1DDEInterfaceError("interface normals must be non-zero")
    cosine = np.einsum("ij,ij->i", n1, dde) / (n1_norm * dde_norm)
    residual = np.abs(1.0 - np.abs(cosine))
    maximum = float(np.max(residual, initial=0.0))
    if maximum > parallel_tolerance:
        raise N1DDEInterfaceError(
            "N1 and DDE interface normals are not parallel"
        )
    alignment = np.where(cosine >= 0.0, 1.0, -1.0)
    return N1DDEOrientation(
        alignment=alignment,
        max_parallel_residual=maximum,
    )


def n1_gamma_to_dde_mu(
    gamma_n1,
    *,
    n1_normal,
    dde_normal,
    parallel_tolerance: float = 1.0e-12,
) -> np.ndarray:
    """Map N1 circulation/potential jump to the DDE scalar convention."""
    gamma = _finite("gamma_n1", gamma_n1, ndim=1)
    orientation = interface_orientation(
        n1_normal,
        dde_normal,
        parallel_tolerance=parallel_tolerance,
    )
    if orientation.alignment.shape != gamma.shape:
        raise N1DDEInterfaceError(
            "one N1 and DDE normal pair is required per circulation value"
        )
    return -orientation.alignment * gamma


def dde_mu_to_n1_gamma(
    mu_dde,
    *,
    n1_normal,
    dde_normal,
    parallel_tolerance: float = 1.0e-12,
) -> np.ndarray:
    """Invert :func:`n1_gamma_to_dde_mu` exactly."""
    mu = _finite("mu_dde", mu_dde, ndim=1)
    orientation = interface_orientation(
        n1_normal,
        dde_normal,
        parallel_tolerance=parallel_tolerance,
    )
    if orientation.alignment.shape != mu.shape:
        raise N1DDEInterfaceError(
            "one N1 and DDE normal pair is required per doublet value"
        )
    return -orientation.alignment * mu

