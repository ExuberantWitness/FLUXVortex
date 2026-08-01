"""Relative IBL flux across a separation curve moving on a material surface.

For a surface conservation law with areal storage ``U`` and tangential flux
``F``, the outward flux through a curve moving at signed co-normal speed
``c_rel`` relative to the material surface is

    (F . nu - c_rel U) ds.

This module applies only that Reynolds-transport identity to the already named
surface-IBL momentum-defect and energy-defect channels.  It does not infer the
curve, its speed, a physical closure, or any newborn VES state.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .surface_ibl_state import (
    SurfaceIBLError,
    SurfaceIBLFields,
    surface_ibl_physical_flux,
)


@dataclass(frozen=True)
class MovingSeparationFlux:
    """Named physical, boundary-motion, and relative IBL outflow rates."""

    physical_momentum_defect_out: np.ndarray
    boundary_motion_momentum_defect_out: np.ndarray
    relative_momentum_defect_out: np.ndarray
    physical_energy_defect_out: np.ndarray
    boundary_motion_energy_defect_out: np.ndarray
    relative_energy_defect_out: np.ndarray
    relative_conormal_speed: np.ndarray
    edge_measure: np.ndarray


def _edge_scalars(name: str, value, count: int) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = np.full(count, float(array))
    if array.shape != (count,):
        raise SurfaceIBLError(
            f"{name} must be scalar or have shape ({count},), got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise SurfaceIBLError(f"{name} contains non-finite values")
    return array.copy()


def moving_separation_ibl_flux(
    fields: SurfaceIBLFields,
    *,
    outward_surface_conormal,
    relative_conormal_speed,
    edge_measure=1.0,
) -> MovingSeparationFlux:
    """Project known IBL storage/flux across an oriented moving curve.

    ``relative_conormal_speed`` is positive when the curve moves along the
    supplied outward co-normal and expands the retained attached region.  The
    resulting ``-c_rel U`` term therefore reduces outward release.
    """
    if not isinstance(fields, SurfaceIBLFields):
        raise SurfaceIBLError("fields must be SurfaceIBLFields")
    speed = _edge_scalars(
        "relative_conormal_speed",
        relative_conormal_speed,
        fields.count,
    )
    measure = _edge_scalars("edge_measure", edge_measure, fields.count)
    if np.any(measure <= 0.0):
        raise SurfaceIBLError("edge_measure must be positive")

    physical = surface_ibl_physical_flux(
        fields,
        outward_surface_conormal=outward_surface_conormal,
        edge_measure=measure,
    )
    boundary_momentum = (
        -speed[:, None]
        * fields.mass_flux_defect
        * measure[:, None]
    )
    boundary_energy = (
        -speed
        * fields.momentum_flux_trace
        * measure
    )
    return MovingSeparationFlux(
        physical_momentum_defect_out=physical.momentum_out,
        boundary_motion_momentum_defect_out=boundary_momentum,
        relative_momentum_defect_out=(
            physical.momentum_out+boundary_momentum
        ),
        physical_energy_defect_out=physical.energy_out,
        boundary_motion_energy_defect_out=boundary_energy,
        relative_energy_defect_out=physical.energy_out+boundary_energy,
        relative_conormal_speed=speed,
        edge_measure=measure,
    )

