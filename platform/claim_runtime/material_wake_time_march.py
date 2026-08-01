"""No-force explicit-stage material-wake time-marching oracle.

This module advances only the actual-boundary/material-wake potential state.
Previously shed P2 rows remain immutable known data.  At every step a
half-time coupled solve generates the explicit middle row of the newborn
band, then a full-time coupled solve generates its current row.  No pressure,
force, LESP, prescribed circulation, regularizer, or target load is present.

The implementation is an equation/time-discretization oracle for
``N3.1j3b6d18b``.  It is not a production aerodynamic solver.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .actual_boundary_body_wake import (
    ActualBoundaryBodyWakeSolution,
    solve_actual_boundary_body_wake_p2,
)
from .classified_p2_cut_topology import ClassifiedP2CutTopology
from .distributed_doublet import (
    MaterialWakeHistory,
    newborn_material_wake_band,
)
from .thick_body_neumann_shadow import (
    ClosedTriangularMesh,
    ThickBodyNeumannError,
)


@dataclass(frozen=True)
class ExplicitMidpointWakeStep:
    """One half-stage/full-stage newborn material-band record."""

    step_index: int
    time_previous: float
    time_midpoint: float
    time_current: float
    body_jump_previous: np.ndarray
    body_jump_midpoint: np.ndarray
    body_jump_current: np.ndarray
    half_stage: ActualBoundaryBodyWakeSolution
    full_stage: ActualBoundaryBodyWakeSolution
    history_after: MaterialWakeHistory
    old_strength_mutation: float
    old_geometry_convection_error: float
    midpoint_row_identity_error: float
    current_attachment_error: float


@dataclass(frozen=True)
class ExplicitMidpointWakeMarch:
    """A fixed-window sequence of explicit midpoint material bands."""

    time_start: float
    time_end: float
    timestep: float
    convection_speed: float
    steps: tuple[ExplicitMidpointWakeStep, ...]
    final_history: MaterialWakeHistory
    final_body_cut_jump: np.ndarray


def _finite_vector(name: str, value: Any, size: int) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ThickBodyNeumannError(
            f"{name} must be finite with shape {(size,)}, got {array.shape}"
        )
    return array.copy()


def _history_rows(
    history: MaterialWakeHistory | None,
    cut_node_count: int,
) -> np.ndarray:
    if history is None:
        return np.empty((0, 3, cut_node_count), dtype=float)
    rows = np.stack(
        [band.potential_jump_rows for band in history.bands],
        axis=0,
    )
    if rows.shape[1:] != (3, cut_node_count):
        raise ThickBodyNeumannError(
            "material history trace does not match body cut"
        )
    return rows


def _convect_history_x(
    history: MaterialWakeHistory | None,
    displacement: float,
) -> MaterialWakeHistory | None:
    if history is None:
        return None
    bands = []
    for band in history.bands:
        vertices = band.surface.vertices.copy()
        vertices[:, 0] += displacement
        bands.append(band.material_update(vertices))
    return MaterialWakeHistory(history.history_id, tuple(bands))


def _maximum_old_state_residuals(
    before: MaterialWakeHistory | None,
    after: MaterialWakeHistory | None,
    displacement: float,
) -> tuple[float, float]:
    if before is None:
        return 0.0, 0.0
    if after is None or len(before.bands) != len(after.bands):
        return np.inf, np.inf
    strength = 0.0
    geometry = 0.0
    for old, moved in zip(before.bands, after.bands):
        strength = max(
            strength,
            float(
                np.max(
                    np.abs(
                        old.potential_jump_rows
                        - moved.potential_jump_rows
                    ),
                    initial=0.0,
                )
            ),
        )
        expected = old.surface.vertices.copy()
        expected[:, 0] += displacement
        geometry = max(
            geometry,
            float(
                np.max(
                    np.linalg.norm(
                        expected - moved.surface.vertices,
                        axis=1,
                    ),
                    initial=0.0,
                )
            ),
        )
    return strength, geometry


def march_actual_boundary_material_wake_explicit_midpoint(
    mesh: ClosedTriangularMesh,
    topology: ClassifiedP2CutTopology,
    *,
    incident_velocity_at_time: Callable[[float], Any],
    initial_body_cut_jump: Any,
    time_start: float,
    time_end: float,
    timestep: float,
    trailing_edge_x: float,
    convection_speed: float,
    target_quadrature_order: int = 10,
    source_quadrature_order: int | None = None,
) -> ExplicitMidpointWakeMarch:
    """Advance a fixed body with a prescribed uniform material convection.

    The half-stage active band uses explicit known rows ``[mu_n, mu_n]`` and
    solves its current row at ``t_n + dt/2``.  The full-stage active band then
    uses ``[mu_n, mu_mid]`` and solves its current row at ``t_n + dt``.
    Consequently the completed band's middle row is a named coupled-stage
    solution, never an average inferred from its endpoints.
    """
    if not callable(incident_velocity_at_time):
        raise ThickBodyNeumannError(
            "incident_velocity_at_time must be callable"
        )
    scalars = np.asarray(
        (
            time_start,
            time_end,
            timestep,
            trailing_edge_x,
            convection_speed,
        ),
        dtype=float,
    )
    if not np.all(np.isfinite(scalars)):
        raise ThickBodyNeumannError("time/convection inputs must be finite")
    if time_end <= time_start or timestep <= 0.0:
        raise ThickBodyNeumannError(
            "time_end must exceed time_start and timestep must be positive"
        )
    if convection_speed <= 0.0:
        raise ThickBodyNeumannError(
            "convection_speed must be strictly positive"
        )
    exact_steps = (time_end - time_start) / timestep
    step_count = int(round(exact_steps))
    if (
        step_count < 1
        or abs(exact_steps - step_count)
        > 128.0 * np.finfo(float).eps * max(abs(exact_steps), 1.0)
    ):
        raise ThickBodyNeumannError(
            "physical time window must contain an integer number of steps"
        )

    cut_node_count = len(topology.cut_node_coordinates)
    current_jump = _finite_vector(
        "initial_body_cut_jump",
        initial_body_cut_jump,
        cut_node_count,
    )
    history: MaterialWakeHistory | None = None
    records: list[ExplicitMidpointWakeStep] = []
    cut_vertices = mesh.vertices[
        topology.ordered_cut_vertex_indices
    ].copy()
    if np.max(
        np.abs(cut_vertices[:, 0] - trailing_edge_x),
        initial=0.0,
    ) > 1.0e-12:
        raise ThickBodyNeumannError(
            "trailing_edge_x is inconsistent with the classified cut"
        )

    for step_index in range(step_count):
        previous_time = time_start + step_index * timestep
        midpoint_time = previous_time + 0.5 * timestep
        current_time = previous_time + timestep
        old_band_count = 0 if history is None else len(history.bands)
        fixed_rows = _history_rows(history, cut_node_count)

        half_edge_x = np.concatenate(
            (
                np.array((trailing_edge_x,)),
                trailing_edge_x
                + convection_speed
                * (
                    0.5 * timestep
                    + timestep * np.arange(old_band_count + 1)
                ),
            )
        )
        half_incident = np.asarray(
            incident_velocity_at_time(midpoint_time),
            dtype=float,
        )
        half = solve_actual_boundary_body_wake_p2(
            mesh,
            topology,
            incident_velocity=half_incident,
            downstream_edge_x=float(half_edge_x[-1]),
            wake_edge_x_nodes=half_edge_x,
            fixed_old_wake_rows=fixed_rows,
            active_known_rows=np.vstack(
                (current_jump, current_jump)
            ),
            target_quadrature_order=target_quadrature_order,
            source_quadrature_order=source_quadrature_order,
        )
        midpoint_jump = half.body_cut_jump.copy()

        full_edge_x = (
            trailing_edge_x
            + convection_speed
            * timestep
            * np.arange(old_band_count + 2)
        )
        full_incident = np.asarray(
            incident_velocity_at_time(current_time),
            dtype=float,
        )
        full = solve_actual_boundary_body_wake_p2(
            mesh,
            topology,
            incident_velocity=full_incident,
            downstream_edge_x=float(full_edge_x[-1]),
            wake_edge_x_nodes=full_edge_x,
            fixed_old_wake_rows=fixed_rows,
            active_known_rows=np.vstack(
                (current_jump, midpoint_jump)
            ),
            target_quadrature_order=target_quadrature_order,
            source_quadrature_order=source_quadrature_order,
        )
        next_jump = full.body_cut_jump.copy()

        convected = _convect_history_x(
            history,
            convection_speed * timestep,
        )
        strength_residual, geometry_residual = (
            _maximum_old_state_residuals(
                history,
                convected,
                convection_speed * timestep,
            )
        )
        previous_edge = cut_vertices.copy()
        previous_edge[:, 0] += convection_speed * timestep
        current_edge = cut_vertices.copy()
        newborn = newborn_material_wake_band(
            sheet_id=f"explicit-midpoint-TEV-{step_index}",
            vortex_family="TEV",
            previous_edge=previous_edge,
            current_edge=current_edge,
            time_nodes=np.array(
                (previous_time, midpoint_time, current_time)
            ),
            potential_jump_rows=np.vstack(
                (current_jump, midpoint_jump, next_jump)
            ),
            span_diagonal_pattern="mirror_symmetric",
        )
        if convected is None:
            history_after = MaterialWakeHistory(
                "explicit-midpoint-TEV-history",
                (newborn,),
            )
        else:
            history_after = convected.append(newborn)

        midpoint_identity = float(
            np.max(
                np.abs(
                    newborn.potential_jump_rows[1]
                    - half.body_cut_jump
                ),
                initial=0.0,
            )
        )
        current_attachment = float(
            np.max(
                np.abs(
                    newborn.potential_jump_rows[2]
                    - full.body_cut_jump
                ),
                initial=0.0,
            )
        )
        records.append(
            ExplicitMidpointWakeStep(
                step_index=step_index,
                time_previous=float(previous_time),
                time_midpoint=float(midpoint_time),
                time_current=float(current_time),
                body_jump_previous=current_jump.copy(),
                body_jump_midpoint=midpoint_jump,
                body_jump_current=next_jump,
                half_stage=half,
                full_stage=full,
                history_after=history_after,
                old_strength_mutation=strength_residual,
                old_geometry_convection_error=geometry_residual,
                midpoint_row_identity_error=midpoint_identity,
                current_attachment_error=current_attachment,
            )
        )
        history = history_after
        current_jump = next_jump

    if history is None:  # pragma: no cover - protected by step_count >= 1
        raise ThickBodyNeumannError("time march produced no material bands")
    return ExplicitMidpointWakeMarch(
        time_start=float(time_start),
        time_end=float(time_end),
        timestep=float(timestep),
        convection_speed=float(convection_speed),
        steps=tuple(records),
        final_history=history,
        final_body_cut_jump=current_jump,
    )

