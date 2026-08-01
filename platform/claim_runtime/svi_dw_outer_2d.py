"""N2.6e1 S0 actual-surface inviscid outer-flow equation oracle.

The solver is the constant-source plus constant-circulation Hess--Smith
construction (Hess & Smith, *Progress in Aeronautical Sciences* 8, 1967,
pp. 1--138) on the actual two-sided NACA wall.  It enforces one normal
boundary equation per panel and one trailing-edge Kutta equation, then
evaluates the exterior tangential trace, pressure coefficient, pressure
traction, and circulation ledger.

This is only an S0 foundation.  Non-zero integral-boundary-layer
transpiration, trailing-edge-wake induction, separated-wake induction,
unsteady Bernoulli pressure, separation selection, and strong iteration are
explicitly rejected.  Reserved arguments expose the future interfaces
without pretending those equations have passed their source-method gates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .svi_dw_types import (
    ActualSurface2D,
    DoubleWakeState2D,
    DualSideIBLState,
    SVIDWFoundationConfig,
    SVIDWFoundationScopeError,
    SVIDWValidationError,
    SurfaceTractionState2D,
    build_naca4_actual_surface,
)


_MAXIMUM_FOUNDATION_CONDITION_NUMBER = 1.0e8


@dataclass(frozen=True)
class HessSmithInfluence2D:
    source_normal: np.ndarray
    source_tangential: np.ndarray
    circulation_normal: np.ndarray
    circulation_tangential: np.ndarray


@dataclass(frozen=True)
class SVIDWOuter2DSolution:
    """Auditable result of the inviscid S0 actual-wall solve."""

    config: SVIDWFoundationConfig
    surface: ActualSurface2D
    freestream_velocity: np.ndarray
    source_strength: np.ndarray
    circulation_sheet_strength: float
    tangential_velocity: np.ndarray
    surface_velocity: np.ndarray
    pressure_coefficient: np.ndarray
    traction: SurfaceTractionState2D
    influence: HessSmithInfluence2D
    system_condition_number: float
    normal_velocity_residual: np.ndarray
    maximum_relative_normal_residual: float
    kutta_residual: float
    relative_kutta_residual: float
    source_flux: float
    relative_source_flux: float
    bound_circulation: float
    surface_circulation: float
    circulation_ledger_absolute_residual: float
    circulation_ledger_residual: float
    force_coefficient_xy: np.ndarray
    drag_coefficient: float
    lift_coefficient: float


def _readonly(name: str, value: Any, shape: tuple[int, ...]) -> np.ndarray:
    array = np.array(value, dtype=float, copy=True)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise SVIDWValidationError(
            f"{name} must be finite with shape {shape}"
        )
    array.setflags(write=False)
    return array


def _panel_local_source_velocity(
    surface: ActualSurface2D,
) -> tuple[np.ndarray, np.ndarray]:
    """Return local (tangent, outward-normal) unit-source velocities."""
    starts = surface.contour_nodes[:-1]
    targets = surface.panel_midpoints
    source_tangent = surface.panel_tangents
    source_normal = surface.panel_outward_normals
    length = surface.panel_lengths
    n_panel = surface.panel_count
    local_tangent = np.empty((n_panel, n_panel), dtype=float)
    local_normal = np.empty_like(local_tangent)
    coefficient = 1.0 / (4.0 * np.pi)

    for source_index in range(n_panel):
        relative = targets - starts[source_index]
        x_local = relative @ source_tangent[source_index]
        y_local = relative @ source_normal[source_index]
        x_after = x_local - length[source_index]
        radius_start_sq = x_local**2 + y_local**2
        radius_end_sq = x_after**2 + y_local**2
        if np.any(radius_start_sq <= 0.0) or np.any(
            radius_end_sq <= 0.0
        ):
            raise SVIDWValidationError(
                "a collocation point coincides with a panel endpoint"
            )
        local_tangent[:, source_index] = coefficient * np.log(
            radius_start_sq / radius_end_sq
        )
        # atan2(cross(r_start,r_end), dot(r_start,r_end)) is a
        # branch-safe signed endpoint angle.  The clockwise surface's left
        # side is exterior, so the self source normal limit is +1/2.
        endpoint_angle = np.arctan2(
            y_local * length[source_index],
            x_local * x_after + y_local**2,
        )
        local_normal[:, source_index] = endpoint_angle / (
            2.0 * np.pi
        )

    diagonal = np.arange(n_panel)
    local_tangent[diagonal, diagonal] = 0.0
    local_normal[diagonal, diagonal] = 0.5
    return local_tangent, local_normal


def hess_smith_influence(
    surface: ActualSurface2D,
) -> HessSmithInfluence2D:
    """Assemble source and unit-uniform-vortex-sheet velocity influences."""
    if not isinstance(surface, ActualSurface2D):
        raise SVIDWValidationError(
            "surface must be an ActualSurface2D"
        )
    source_local_tangent, source_local_normal = (
        _panel_local_source_velocity(surface)
    )
    target_tangent = surface.panel_tangents
    target_normal = surface.panel_outward_normals
    source_tangent = surface.panel_tangents
    source_normal = surface.panel_outward_normals

    tangent_dot_tangent = target_tangent @ source_tangent.T
    tangent_dot_normal = target_tangent @ source_normal.T
    normal_dot_tangent = target_normal @ source_tangent.T
    normal_dot_normal = target_normal @ source_normal.T

    source_tangential = (
        source_local_tangent * tangent_dot_tangent
        + source_local_normal * tangent_dot_normal
    )
    source_normal_influence = (
        source_local_tangent * normal_dot_tangent
        + source_local_normal * normal_dot_normal
    )

    # A positive point-vortex sheet is the source velocity rotated +90 deg:
    # (u_t,u_n)_vortex = (-u_n,u_t)_source.
    vortex_tangential_each = (
        -source_local_normal * tangent_dot_tangent
        + source_local_tangent * tangent_dot_normal
    )
    vortex_normal_each = (
        -source_local_normal * normal_dot_tangent
        + source_local_tangent * normal_dot_normal
    )
    circulation_tangential = np.sum(
        vortex_tangential_each, axis=1
    )
    circulation_normal = np.sum(vortex_normal_each, axis=1)
    n_panel = surface.panel_count
    return HessSmithInfluence2D(
        source_normal=_readonly(
            "source_normal", source_normal_influence, (n_panel, n_panel)
        ),
        source_tangential=_readonly(
            "source_tangential",
            source_tangential,
            (n_panel, n_panel),
        ),
        circulation_normal=_readonly(
            "circulation_normal", circulation_normal, (n_panel,)
        ),
        circulation_tangential=_readonly(
            "circulation_tangential",
            circulation_tangential,
            (n_panel,),
        ),
    )


def _validate_reserved_zero_interfaces(
    surface: ActualSurface2D,
    *,
    ibl_state: DualSideIBLState | None,
    double_wake_state: DoubleWakeState2D | None,
    transpiration_velocity: Any | None,
) -> None:
    if ibl_state is not None:
        if not isinstance(ibl_state, DualSideIBLState):
            raise SVIDWValidationError(
                "ibl_state must be a DualSideIBLState"
            )
        expected = (2, surface.panels_per_side)
        if ibl_state.displacement_thickness.shape != expected:
            raise SVIDWValidationError(
                f"ibl_state arrays must have shape {expected}"
            )
        if ibl_state.has_viscous_coupling:
            raise SVIDWFoundationScopeError(
                "active IBL/transpiration requires the later N2.6e1 "
                "strong-coupling milestone"
            )
    if double_wake_state is not None:
        if not isinstance(double_wake_state, DoubleWakeState2D):
            raise SVIDWValidationError(
                "double_wake_state must be a DoubleWakeState2D"
            )
        if double_wake_state.has_induction:
            raise SVIDWFoundationScopeError(
                "wake induction/evolution is not implemented in N2.6e1 S0"
            )
    if transpiration_velocity is not None:
        transpiration = np.asarray(
            transpiration_velocity, dtype=float
        )
        if (
            transpiration.shape != (surface.panel_count,)
            or not np.all(np.isfinite(transpiration))
        ):
            raise SVIDWValidationError(
                "transpiration_velocity must be finite with one value "
                "per contour panel"
            )
        if np.any(transpiration):
            raise SVIDWFoundationScopeError(
                "non-zero transpiration is reserved for the later "
                "N2.6e1 strong-coupling milestone"
            )


def solve_svi_dw_outer_foundation(
    config: SVIDWFoundationConfig,
    *,
    surface: ActualSurface2D | None = None,
    ibl_state: DualSideIBLState | None = None,
    double_wake_state: DoubleWakeState2D | None = None,
    transpiration_velocity: Any | None = None,
) -> SVIDWOuter2DSolution:
    """Solve the no-deficit/no-wake N2.6e1 S0 outer-flow foundation."""
    if not isinstance(config, SVIDWFoundationConfig):
        raise SVIDWValidationError(
            "config must be a SVIDWFoundationConfig"
        )
    if surface is None:
        wall = build_naca4_actual_surface(
            config.section,
            panels_per_side=config.panels_per_side,
        )
    else:
        if not isinstance(surface, ActualSurface2D):
            raise SVIDWValidationError(
                "surface must be an ActualSurface2D"
            )
        if surface.section != config.section:
            raise SVIDWValidationError(
                "surface section does not match configuration"
            )
        if surface.panels_per_side != config.panels_per_side:
            raise SVIDWValidationError(
                "surface panel count does not match configuration"
            )
        wall = surface
    _validate_reserved_zero_interfaces(
        wall,
        ibl_state=ibl_state,
        double_wake_state=double_wake_state,
        transpiration_velocity=transpiration_velocity,
    )

    alpha = np.deg2rad(config.angle_of_attack_deg)
    flow_direction = np.array(
        [np.cos(alpha), np.sin(alpha)], dtype=float
    )
    lift_direction = np.array(
        [-np.sin(alpha), np.cos(alpha)], dtype=float
    )
    freestream = config.freestream_speed * flow_direction
    influence = hess_smith_influence(wall)
    n_panel = wall.panel_count
    system = np.empty((n_panel + 1, n_panel + 1), dtype=float)
    rhs = np.empty(n_panel + 1, dtype=float)
    system[:n_panel, :n_panel] = influence.source_normal
    system[:n_panel, -1] = influence.circulation_normal
    rhs[:n_panel] = -(wall.panel_outward_normals @ freestream)

    # For a cusp represented by the first lower and last upper panel, finite
    # velocity requires equal-and-opposite contour tangential traces.
    lower_te = 0
    upper_te = n_panel - 1
    system[-1, :n_panel] = (
        influence.source_tangential[lower_te]
        + influence.source_tangential[upper_te]
    )
    system[-1, -1] = (
        influence.circulation_tangential[lower_te]
        + influence.circulation_tangential[upper_te]
    )
    rhs[-1] = -float(
        freestream
        @ (
            wall.panel_tangents[lower_te]
            + wall.panel_tangents[upper_te]
        )
    )
    condition_number = float(np.linalg.cond(system))
    if not np.isfinite(condition_number):
        raise SVIDWValidationError(
            "Hess-Smith linear system is singular or non-finite"
        )
    if condition_number > _MAXIMUM_FOUNDATION_CONDITION_NUMBER:
        raise SVIDWValidationError(
            "Hess-Smith foundation is too ill-conditioned "
            f"(condition number {condition_number:.6e}); near-coincident "
            "two-sided geometry is not a valid thin-sheet limit. Use a "
            "dedicated thin-sheet formulation instead"
        )
    try:
        unknown = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError as error:
        raise SVIDWValidationError(
            "Hess-Smith linear solve failed"
        ) from error
    source_strength = unknown[:-1]
    circulation_strength = float(unknown[-1])

    normal_residual = (
        wall.panel_outward_normals @ freestream
        + influence.source_normal @ source_strength
        + influence.circulation_normal * circulation_strength
    )
    tangential_velocity = (
        wall.panel_tangents @ freestream
        + influence.source_tangential @ source_strength
        + influence.circulation_tangential * circulation_strength
    )
    pressure_coefficient = 1.0 - (
        tangential_velocity / config.freestream_speed
    ) ** 2
    surface_velocity = (
        tangential_velocity[:, None] * wall.panel_tangents
    )
    dynamic_pressure = (
        0.5 * config.density * config.freestream_speed**2
    )
    traction = SurfaceTractionState2D.from_pressure_and_shear(
        wall,
        pressure=dynamic_pressure * pressure_coefficient,
        wall_shear=np.zeros(n_panel, dtype=float),
    )
    force_coefficient = (
        traction.resultant_force
        / (dynamic_pressure * config.section.chord)
    )
    drag_coefficient = float(force_coefficient @ flow_direction)
    lift_coefficient = float(force_coefficient @ lift_direction)

    velocity_scale = max(
        config.freestream_speed, np.finfo(float).tiny
    )
    maximum_relative_normal_residual = float(
        np.max(np.abs(normal_residual), initial=0.0) / velocity_scale
    )
    kutta_residual = float(
        tangential_velocity[lower_te]
        + tangential_velocity[upper_te]
    )
    relative_kutta_residual = abs(kutta_residual) / velocity_scale
    source_flux = float(source_strength @ wall.panel_lengths)
    source_flux_scale = max(
        float(np.abs(source_strength) @ wall.panel_lengths),
        np.finfo(float).tiny,
    )
    relative_source_flux = abs(source_flux) / source_flux_scale

    # For the clockwise contour and positive point-vortex convention used in
    # ``hess_smith_influence``, the exterior trace of a uniform vortex sheet
    # carries -gamma per unit arc length.
    bound_circulation = (
        -circulation_strength * wall.perimeter
    )
    surface_circulation = float(
        tangential_velocity @ wall.panel_lengths
    )
    circulation_ledger_absolute_residual = (
        surface_circulation - bound_circulation
    )
    circulation_ledger_residual = (
        circulation_ledger_absolute_residual
        / (config.freestream_speed * config.section.chord)
    )

    return SVIDWOuter2DSolution(
        config=config,
        surface=wall,
        freestream_velocity=_readonly(
            "freestream_velocity", freestream, (2,)
        ),
        source_strength=_readonly(
            "source_strength", source_strength, (n_panel,)
        ),
        circulation_sheet_strength=circulation_strength,
        tangential_velocity=_readonly(
            "tangential_velocity", tangential_velocity, (n_panel,)
        ),
        surface_velocity=_readonly(
            "surface_velocity", surface_velocity, (n_panel, 2)
        ),
        pressure_coefficient=_readonly(
            "pressure_coefficient", pressure_coefficient, (n_panel,)
        ),
        traction=traction,
        influence=influence,
        system_condition_number=condition_number,
        normal_velocity_residual=_readonly(
            "normal_velocity_residual", normal_residual, (n_panel,)
        ),
        maximum_relative_normal_residual=(
            maximum_relative_normal_residual
        ),
        kutta_residual=kutta_residual,
        relative_kutta_residual=relative_kutta_residual,
        source_flux=source_flux,
        relative_source_flux=relative_source_flux,
        bound_circulation=bound_circulation,
        surface_circulation=surface_circulation,
        circulation_ledger_absolute_residual=float(
            circulation_ledger_absolute_residual
        ),
        circulation_ledger_residual=float(
            circulation_ledger_residual
        ),
        force_coefficient_xy=_readonly(
            "force_coefficient_xy", force_coefficient, (2,)
        ),
        drag_coefficient=drag_coefficient,
        lift_coefficient=lift_coefficient,
    )
