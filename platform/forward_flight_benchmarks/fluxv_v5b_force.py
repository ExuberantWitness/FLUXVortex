"""Single-owner surface-pressure force ledger for the FluxV v5b shadow.

This module is deliberately narrower than a solver adapter.  It observes one
already-completed :class:`HiratoLiveStepReport` and converts that shared
bound/TEV/LEV state to panel pressure and force exactly once.  It does not
advance a wake, solve circulation, add an LDVM increment, or provide an
impulse/polar force.

The pressure convention is the repository's unified potential-jump identity::

    delta_p = rho * (V_local . grad_s(mu) + d(mu)/dt)

where ``mu = Gamma_b + Gamma_L`` on actively shedding strips.  Hirato Eq.17
owns the split between the bound and LEV time-rate channels.  The no-LEV path
delegates to the same N1 baseline assembly, making reduction an exact code
path rather than a numerical coincidence.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np

from claim_runtime.hirato_equations import PotentialRate, potential_rate_eq17
from claim_runtime.hirato_live_shadow import BoundLattice, HiratoLiveStepReport
from claim_runtime.hirato_shadow import (
    mirrored_ring_field,
    ring_field_velocity_lamb_oseen,
)
from claim_runtime.unified_panel_pressure import (
    UnifiedPanelPressure,
    UnifiedPressureError,
    structured_uvlm_surface_gradient,
    unified_panel_pressure,
)


class FluxVV5BForceError(ValueError):
    """The live state is insufficient or inconsistent for a unique force."""


def _finite(name: str, value, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise FluxVV5BForceError(
            f"{name} must be finite with shape {shape}, got {array.shape}"
        )
    return array


def _readonly(value) -> np.ndarray:
    array = np.array(value, dtype=float, copy=True)
    array.setflags(write=False)
    return array


def _readonly_mapping(values: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
    return MappingProxyType({name: _readonly(value) for name, value in values.items()})


@dataclass(frozen=True)
class PanelSurfaceGeometry:
    """Structured UVLM geometry reconstructed from bound vortex rings."""

    area: np.ndarray
    chord_tangent: np.ndarray
    span_tangent: np.ndarray
    chord_step: np.ndarray
    span_step: np.ndarray


@dataclass(frozen=True)
class FluxVV5BForceGuards:
    velocity_ledger_residual: float
    pressure_ledger_residual: float
    force_channel_residual: float
    total_force_residual: float
    total_moment_residual: float
    no_lev_exact_reduction_required: bool
    no_lev_exact_reduction_passed: bool
    all_finite: bool

    @property
    def passed(self) -> bool:
        return bool(
            self.all_finite
            and self.velocity_ledger_residual <= 1.0e-12
            and self.pressure_ledger_residual <= 1.0e-12
            and self.force_channel_residual <= 1.0e-12
            and self.total_force_residual <= 1.0e-12
            and self.total_moment_residual <= 1.0e-12
            and (
                not self.no_lev_exact_reduction_required
                or self.no_lev_exact_reduction_passed
            )
        )


@dataclass(frozen=True)
class FluxVV5BForceLedger:
    """One immutable panel-pressure/force observation of a live shadow step."""

    pressure: UnifiedPanelPressure
    geometry: PanelSurfaceGeometry
    local_velocity: np.ndarray
    local_velocity_channels: Mapping[str, np.ndarray]
    surface_potential_jump: np.ndarray
    surface_gradient: np.ndarray
    potential_rate: PotentialRate
    panel_force: np.ndarray
    panel_moment: np.ndarray
    total_force: np.ndarray
    total_moment: np.ndarray
    moment_origin: np.ndarray
    guards: FluxVV5BForceGuards


def reconstruct_panel_surface_geometry(
    lattice: BoundLattice,
    *,
    nc: int,
    ns: int,
) -> PanelSurfaceGeometry:
    """Reconstruct the frozen structured pressure stencil geometry."""
    count = int(nc) * int(ns)
    if nc <= 0 or ns <= 0:
        raise FluxVV5BForceError("nc and ns must be positive")
    rings = _finite("lattice.rings", lattice.rings, (count, 4, 3))
    chord_vector = 0.5 * ((rings[:, 2] - rings[:, 0]) + (rings[:, 3] - rings[:, 1]))
    span_vector = 0.5 * ((rings[:, 1] - rings[:, 0]) + (rings[:, 2] - rings[:, 3]))
    chord_step = np.linalg.norm(chord_vector, axis=1)
    span_step = np.linalg.norm(span_vector, axis=1)
    if np.any(chord_step <= 0.0) or np.any(span_step <= 0.0):
        raise FluxVV5BForceError("bound lattice has a zero surface step")
    chord_tangent = chord_vector / chord_step[:, None]
    span_tangent = span_vector / span_step[:, None]
    area = 0.5 * np.linalg.norm(
        np.cross(rings[:, 2] - rings[:, 0], rings[:, 3] - rings[:, 1]),
        axis=1,
    )
    if np.any(area <= 0.0):
        raise FluxVV5BForceError("bound lattice has a zero panel area")
    return PanelSurfaceGeometry(
        area=_readonly(area),
        chord_tangent=_readonly(chord_tangent),
        span_tangent=_readonly(span_tangent),
        chord_step=_readonly(chord_step),
        span_step=_readonly(span_step),
    )


def _ring_velocity(
    points: np.ndarray,
    rings,
    gamma,
    *,
    core_radius: float,
    mirror_symmetry: bool,
) -> np.ndarray:
    geometry = np.asarray(rings, dtype=float)
    strength = np.asarray(gamma, dtype=float)
    if geometry.shape == (0, 4, 3) and strength.shape == (0,):
        return np.zeros_like(points)
    if geometry.ndim != 3 or geometry.shape[1:] != (4, 3):
        raise FluxVV5BForceError(
            f"ring field must have shape (n,4,3), got {geometry.shape}"
        )
    if strength.shape != (len(geometry),):
        raise FluxVV5BForceError(
            f"ring strength must have shape {(len(geometry),)}, got {strength.shape}"
        )
    if not np.all(np.isfinite(geometry)) or not np.all(np.isfinite(strength)):
        raise FluxVV5BForceError("ring field contains non-finite data")
    velocity = ring_field_velocity_lamb_oseen(points, geometry, strength, core_radius)
    if mirror_symmetry:
        velocity = velocity + ring_field_velocity_lamb_oseen(
            points,
            mirrored_ring_field(geometry),
            strength,
            core_radius,
        )
    return velocity


def _validate_state(
    lattice: BoundLattice,
    report: HiratoLiveStepReport,
    *,
    nc: int,
    ns: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = int(nc) * int(ns)
    collocation = _finite("lattice.collocation", lattice.collocation, (count, 3))
    normals = _finite("lattice.normals", lattice.normals, (count, 3))
    velocity = _finite(
        "lattice.collocation_velocity",
        lattice.collocation_velocity,
        (count, 3),
    )
    _finite("report.bound_gamma", report.bound_gamma, (count,))
    report_rings = _finite("report.bound_rings", report.bound_rings, (count, 4, 3))
    lattice_rings = _finite("lattice.rings", lattice.rings, (count, 4, 3))
    if not np.array_equal(report_rings, lattice_rings):
        raise FluxVV5BForceError(
            "report.bound_rings must be the exact BoundLattice used by the solve"
        )
    active = np.asarray(report.active, dtype=bool)
    if active.shape != (ns,):
        raise FluxVV5BForceError(
            f"report.active must have shape {(ns,)}, got {active.shape}"
        )
    gamma_lev = _finite("report.gamma_lev", report.gamma_lev, (ns,))
    if np.any(~active & (gamma_lev != 0.0)):
        raise FluxVV5BForceError("inactive strips must have zero current gamma_lev")
    normal_norm = np.linalg.norm(normals, axis=1)
    if np.max(np.abs(normal_norm - 1.0), initial=0.0) > 1.0e-10:
        raise FluxVV5BForceError("lattice normals must be unit length")
    return collocation, normals, velocity


def _velocity_channels(
    lattice: BoundLattice,
    report: HiratoLiveStepReport,
    *,
    u_infinity: np.ndarray,
    core_radius: float,
    mirror_symmetry: bool,
) -> Mapping[str, np.ndarray]:
    points = np.asarray(lattice.collocation, dtype=float)
    freestream_motion = u_infinity[None, :] - np.asarray(
        lattice.collocation_velocity, dtype=float
    )
    pseudo_gamma = np.asarray(report.gamma_lev, dtype=float)
    bound = _ring_velocity(
        points,
        report.bound_rings,
        report.bound_gamma,
        core_radius=core_radius,
        mirror_symmetry=mirror_symmetry,
    )
    if np.any(pseudo_gamma):
        bound = bound + _ring_velocity(
            points,
            report.pseudo_rings,
            pseudo_gamma,
            core_radius=core_radius,
            mirror_symmetry=mirror_symmetry,
        )
    tev = _ring_velocity(
        points,
        report.tev_pre_convection.rings,
        report.tev_pre_convection.gamma,
        core_radius=core_radius,
        mirror_symmetry=mirror_symmetry,
    )
    lev = _ring_velocity(
        points,
        report.lev_pre_convection.rings,
        report.lev_pre_convection.gamma,
        core_radius=core_radius,
        mirror_symmetry=mirror_symmetry,
    )
    return _readonly_mapping(
        {
            "freestream_motion": freestream_motion,
            "bound": bound,
            "tev": tev,
            "lev": lev,
        }
    )


def _baseline_force(
    *,
    density: float,
    dt: float,
    nc: int,
    ns: int,
    lattice: BoundLattice,
    report: HiratoLiveStepReport,
    previous_bound_gamma: np.ndarray,
    u_infinity: np.ndarray,
    core_radius: float,
    mirror_symmetry: bool,
    moment_origin: np.ndarray,
) -> FluxVV5BForceLedger:
    """Assemble the same N1 pressure baseline, with TEV but no LEV state."""
    count = int(nc) * int(ns)
    geometry = reconstruct_panel_surface_geometry(lattice, nc=nc, ns=ns)
    channels = _velocity_channels(
        lattice,
        report,
        u_infinity=u_infinity,
        core_radius=core_radius,
        mirror_symmetry=mirror_symmetry,
    )
    # A no-LEV report gives an identically zero LE channel.  Refuse to call
    # this baseline with hidden LE inventory instead of silently discarding it.
    if len(report.lev_pre_convection.rings) or np.any(report.gamma_lev):
        raise FluxVV5BForceError("N1 no-LEV baseline received LEV inventory")
    local_velocity = np.sum(np.stack(tuple(channels.values())), axis=0)
    surface_jump = np.asarray(report.bound_gamma, dtype=float)
    gradient = structured_uvlm_surface_gradient(
        surface_jump,
        chord_tangent=geometry.chord_tangent,
        span_tangent=geometry.span_tangent,
        chord_step=geometry.chord_step,
        span_step=geometry.span_step,
        nc=nc,
        ns=ns,
    )
    zero_lev = np.zeros(ns)
    rate = potential_rate_eq17(
        surface_jump.reshape(nc, ns),
        previous_bound_gamma.reshape(nc, ns),
        zero_lev,
        zero_lev,
        np.zeros(ns, dtype=bool),
        dt,
    )
    pressure = unified_panel_pressure(
        density=density,
        local_velocity=local_velocity,
        surface_gradient=gradient,
        potential_rate_channels={"bound_unsteady": rate.bound.reshape(count)},
        area=geometry.area,
        normal=lattice.normals,
    )
    return _finalize_ledger(
        pressure=pressure,
        geometry=geometry,
        channels=channels,
        local_velocity=local_velocity,
        surface_jump=surface_jump,
        gradient=gradient,
        rate=rate,
        collocation=lattice.collocation,
        moment_origin=moment_origin,
        no_lev_required=True,
        no_lev_passed=True,
    )


def _finalize_ledger(
    *,
    pressure: UnifiedPanelPressure,
    geometry: PanelSurfaceGeometry,
    channels: Mapping[str, np.ndarray],
    local_velocity: np.ndarray,
    surface_jump: np.ndarray,
    gradient: np.ndarray,
    rate: PotentialRate,
    collocation: np.ndarray,
    moment_origin: np.ndarray,
    no_lev_required: bool,
    no_lev_passed: bool,
) -> FluxVV5BForceLedger:
    panel_force = np.asarray(pressure.total_force, dtype=float)
    arms = np.asarray(collocation, dtype=float) - moment_origin[None, :]
    panel_moment = np.cross(arms, panel_force)
    total_force = np.sum(panel_force, axis=0)
    total_moment = np.sum(panel_moment, axis=0)
    velocity_sum = np.sum(np.stack(tuple(channels.values())), axis=0)
    pressure_report = pressure.ledger_report()
    channel_force = np.sum(np.stack(tuple(pressure.force_channels.values())), axis=0)
    guards = FluxVV5BForceGuards(
        velocity_ledger_residual=float(
            np.max(np.abs(velocity_sum - local_velocity), initial=0.0)
        ),
        pressure_ledger_residual=float(pressure_report.pressure_residual),
        force_channel_residual=float(
            np.max(np.abs(channel_force - panel_force), initial=0.0)
        ),
        total_force_residual=float(
            np.max(np.abs(np.sum(panel_force, axis=0) - total_force), initial=0.0)
        ),
        total_moment_residual=float(
            np.max(np.abs(np.sum(panel_moment, axis=0) - total_moment), initial=0.0)
        ),
        no_lev_exact_reduction_required=bool(no_lev_required),
        no_lev_exact_reduction_passed=bool(no_lev_passed),
        all_finite=bool(
            np.all(np.isfinite(panel_force))
            and np.all(np.isfinite(panel_moment))
            and np.all(np.isfinite(local_velocity))
        ),
    )
    if not guards.passed:
        raise FluxVV5BForceError(f"surface-force ledger failed: {guards}")
    return FluxVV5BForceLedger(
        pressure=pressure,
        geometry=geometry,
        local_velocity=_readonly(local_velocity),
        local_velocity_channels=channels,
        surface_potential_jump=_readonly(surface_jump),
        surface_gradient=_readonly(gradient),
        potential_rate=PotentialRate(
            bound=_readonly(rate.bound),
            lev=_readonly(rate.lev),
            total=_readonly(rate.total),
        ),
        panel_force=_readonly(panel_force),
        panel_moment=_readonly(panel_moment),
        total_force=_readonly(total_force),
        total_moment=_readonly(total_moment),
        moment_origin=_readonly(moment_origin),
        guards=guards,
    )


def n1_no_lev_pressure_baseline(
    *,
    density: float,
    dt: float,
    nc: int,
    ns: int,
    lattice: BoundLattice,
    report: HiratoLiveStepReport,
    previous_bound_gamma,
    u_infinity,
    core_radius: float,
    mirror_symmetry: bool,
    moment_origin=(0.0, 0.0, 0.0),
) -> FluxVV5BForceLedger:
    """Public exact-reduction baseline for a report with no LEV inventory."""
    if not np.isfinite(density) or density <= 0.0:
        raise FluxVV5BForceError("density must be positive and finite")
    if not np.isfinite(dt) or dt <= 0.0:
        raise FluxVV5BForceError("dt must be positive and finite")
    if not np.isfinite(core_radius) or core_radius <= 0.0:
        raise FluxVV5BForceError("core_radius must be positive and finite")
    _validate_state(lattice, report, nc=nc, ns=ns)
    count = int(nc) * int(ns)
    previous = _finite("previous_bound_gamma", previous_bound_gamma, (count,))
    freestream = _finite("u_infinity", u_infinity, (3,))
    origin = _finite("moment_origin", moment_origin, (3,))
    try:
        return _baseline_force(
            density=float(density),
            dt=float(dt),
            nc=int(nc),
            ns=int(ns),
            lattice=lattice,
            report=report,
            previous_bound_gamma=previous,
            u_infinity=freestream,
            core_radius=float(core_radius),
            mirror_symmetry=bool(mirror_symmetry),
            moment_origin=origin,
        )
    except UnifiedPressureError as error:
        raise FluxVV5BForceError(str(error)) from error


def fluxv_v5b_surface_force(
    *,
    density: float,
    dt: float,
    nc: int,
    ns: int,
    lattice: BoundLattice,
    report: HiratoLiveStepReport,
    previous_bound_gamma,
    previous_gamma_lev,
    u_infinity,
    core_radius: float,
    mirror_symmetry: bool,
    moment_origin=(0.0, 0.0, 0.0),
) -> FluxVV5BForceLedger:
    """Assemble the unique v5b surface-pressure force for one live step."""
    if not np.isfinite(density) or density <= 0.0:
        raise FluxVV5BForceError("density must be positive and finite")
    if not np.isfinite(dt) or dt <= 0.0:
        raise FluxVV5BForceError("dt must be positive and finite")
    if not np.isfinite(core_radius) or core_radius <= 0.0:
        raise FluxVV5BForceError("core_radius must be positive and finite")
    _validate_state(lattice, report, nc=nc, ns=ns)
    count = int(nc) * int(ns)
    previous_bound = _finite("previous_bound_gamma", previous_bound_gamma, (count,))
    previous_lev = _finite("previous_gamma_lev", previous_gamma_lev, (ns,))
    freestream = _finite("u_infinity", u_infinity, (3,))
    origin = _finite("moment_origin", moment_origin, (3,))

    no_lev = bool(
        len(report.lev_pre_convection.rings) == 0
        and not np.any(report.gamma_lev)
        and not np.any(previous_lev)
    )
    if no_lev:
        # This is the exact same function call returned by the public baseline.
        return _baseline_force(
            density=float(density),
            dt=float(dt),
            nc=int(nc),
            ns=int(ns),
            lattice=lattice,
            report=report,
            previous_bound_gamma=previous_bound,
            u_infinity=freestream,
            core_radius=float(core_radius),
            mirror_symmetry=bool(mirror_symmetry),
            moment_origin=origin,
        )

    geometry = reconstruct_panel_surface_geometry(lattice, nc=nc, ns=ns)
    channels = _velocity_channels(
        lattice,
        report,
        u_infinity=freestream,
        core_radius=float(core_radius),
        mirror_symmetry=bool(mirror_symmetry),
    )
    local_velocity = np.sum(np.stack(tuple(channels.values())), axis=0)
    release_current = np.where(
        np.asarray(report.active, dtype=bool),
        np.asarray(report.gamma_lev, dtype=float),
        0.0,
    )
    release_panel = np.tile(release_current, int(nc))
    surface_jump = np.asarray(report.bound_gamma, dtype=float) + release_panel
    gradient = structured_uvlm_surface_gradient(
        surface_jump,
        chord_tangent=geometry.chord_tangent,
        span_tangent=geometry.span_tangent,
        chord_step=geometry.chord_step,
        span_step=geometry.span_step,
        nc=int(nc),
        ns=int(ns),
    )
    rate = potential_rate_eq17(
        np.asarray(report.bound_gamma, dtype=float).reshape(nc, ns),
        previous_bound.reshape(nc, ns),
        np.asarray(report.gamma_lev, dtype=float),
        previous_lev,
        np.asarray(report.active, dtype=bool),
        float(dt),
    )
    try:
        pressure = unified_panel_pressure(
            density=float(density),
            local_velocity=local_velocity,
            surface_gradient=gradient,
            potential_rate_channels={
                "bound_unsteady": rate.bound.reshape(count),
                "lev_sheet_unsteady": rate.lev.reshape(count),
            },
            area=geometry.area,
            normal=lattice.normals,
        )
    except UnifiedPressureError as error:
        raise FluxVV5BForceError(str(error)) from error
    return _finalize_ledger(
        pressure=pressure,
        geometry=geometry,
        channels=channels,
        local_velocity=local_velocity,
        surface_jump=surface_jump,
        gradient=gradient,
        rate=rate,
        collocation=lattice.collocation,
        moment_origin=origin,
        no_lev_required=False,
        no_lev_passed=True,
    )


__all__ = [
    "FluxVV5BForceError",
    "FluxVV5BForceGuards",
    "FluxVV5BForceLedger",
    "PanelSurfaceGeometry",
    "fluxv_v5b_surface_force",
    "n1_no_lev_pressure_baseline",
    "reconstruct_panel_surface_geometry",
]
