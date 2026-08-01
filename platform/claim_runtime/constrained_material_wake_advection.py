"""Body-attached, no-force material-wake Heun geometry semantics.

Free material vertices use an explicit velocity provider.  The newest
current edge is an essential aerodynamic-kinematics boundary supplied by the
body cut at each stage.  Potential-jump rows remain immutable, and duplicate
chronological seams must receive the same velocity rather than being averaged.

This module contains no bound-strength solve, pressure, force, LESP, core,
smoothing, structural dynamics or target load.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .distributed_doublet import (
    DistributedDoubletError,
    MaterialWakeHistory,
    MaterialWakeHistoryReport,
)


HistoryVelocityProvider = Callable[
    [MaterialWakeHistory, float],
    tuple[np.ndarray, ...],
]
AttachedEdgeProvider = Callable[[float], np.ndarray]


@dataclass(frozen=True)
class ConstrainedWakeAdvectionReport:
    dt: float
    stage0_seam_velocity_error: float
    stage1_seam_velocity_error: float
    attached_edge_error: float
    material_strength_residual: float
    minimum_face_area_ratio: float
    history: MaterialWakeHistoryReport
    passed: bool


@dataclass(frozen=True)
class ConstrainedWakeAdvectionStep:
    history: MaterialWakeHistory
    report: ConstrainedWakeAdvectionReport


def _velocities(
    history: MaterialWakeHistory,
    time: float,
    provider: HistoryVelocityProvider,
) -> tuple[np.ndarray, ...]:
    raw = provider(history, float(time))
    if not isinstance(raw, tuple) or len(raw) != len(history.bands):
        raise DistributedDoubletError(
            "history velocity provider must return one array per band"
        )
    output = []
    for band, value in zip(history.bands, raw):
        velocity = np.asarray(value, dtype=float)
        if (
            velocity.shape != band.surface.vertices.shape
            or not np.all(np.isfinite(velocity))
        ):
            raise DistributedDoubletError(
                "history velocity has incompatible shape or non-finite data"
            )
        output.append(velocity.copy())
    return tuple(output)


def _seam_velocity_error(
    history: MaterialWakeHistory,
    velocities: tuple[np.ndarray, ...],
) -> float:
    maximum = 0.0
    for older, newer, old_velocity, new_velocity in zip(
        history.bands,
        history.bands[1:],
        velocities,
        velocities[1:],
    ):
        maximum = max(
            maximum,
            float(
                np.max(
                    np.linalg.norm(
                        old_velocity[older.span_nodes :]
                        - new_velocity[: newer.span_nodes],
                        axis=1,
                    ),
                    initial=0.0,
                )
            ),
        )
    return maximum


def _attached_edge(
    history: MaterialWakeHistory,
    time: float,
    provider: AttachedEdgeProvider,
) -> np.ndarray:
    edge = np.asarray(provider(float(time)), dtype=float)
    expected = (
        history.bands[-1].span_nodes,
        3,
    )
    if edge.shape != expected or not np.all(np.isfinite(edge)):
        raise DistributedDoubletError(
            f"attached edge provider must return shape {expected}"
        )
    return edge.copy()


def _update_history(
    history: MaterialWakeHistory,
    displacements: tuple[np.ndarray, ...],
    attached_edge: np.ndarray,
) -> MaterialWakeHistory:
    bands = []
    last = len(history.bands) - 1
    for index, (band, displacement) in enumerate(
        zip(history.bands, displacements)
    ):
        vertices = band.surface.vertices + displacement
        if index == last:
            vertices[band.span_nodes :] = attached_edge
        bands.append(band.material_update(vertices))
    candidate = MaterialWakeHistory(
        history.history_id,
        tuple(bands),
    )
    report = candidate.continuity_report()
    if not report.compatible:
        raise DistributedDoubletError(
            f"constrained wake update broke material seams: {report}"
        )
    return candidate


def _minimum_area_ratio(
    before: MaterialWakeHistory,
    after: MaterialWakeHistory,
) -> float:
    minimum = np.inf
    for old_band, new_band in zip(before.bands, after.bands):
        for face_index in range(len(old_band.surface)):
            old_area = old_band.surface.element(face_index).area
            new_area = new_band.surface.element(face_index).area
            minimum = min(minimum, new_area / old_area)
    return float(minimum)


def advance_constrained_material_wake_heun(
    history: MaterialWakeHistory,
    *,
    time: float,
    dt: float,
    velocity_provider: HistoryVelocityProvider,
    attached_edge_provider: AttachedEdgeProvider,
    seam_velocity_tolerance: float = 1.0e-12,
    attachment_tolerance: float = 1.0e-12,
    strength_tolerance: float = 1.0e-12,
    minimum_face_area_ratio: float = 0.0,
) -> ConstrainedWakeAdvectionStep:
    """Advance free vertices with Heun and clamp the newest body edge."""
    if not isinstance(history, MaterialWakeHistory):
        raise DistributedDoubletError(
            "history must be MaterialWakeHistory"
        )
    scalars = (
        time,
        dt,
        seam_velocity_tolerance,
        attachment_tolerance,
        strength_tolerance,
        minimum_face_area_ratio,
    )
    if not all(np.isfinite(value) for value in scalars):
        raise DistributedDoubletError(
            "time, dt and tolerances must be finite"
        )
    if dt <= 0.0:
        raise DistributedDoubletError("dt must be positive")
    if any(
        value < 0.0
        for value in (
            seam_velocity_tolerance,
            attachment_tolerance,
            strength_tolerance,
            minimum_face_area_ratio,
        )
    ):
        raise DistributedDoubletError(
            "advection tolerances must be non-negative"
        )
    if not callable(velocity_provider) or not callable(
        attached_edge_provider
    ):
        raise DistributedDoubletError(
            "velocity and attached-edge providers must be callable"
        )
    current_edge = _attached_edge(
        history,
        time,
        attached_edge_provider,
    )
    current_actual = history.bands[-1].surface.vertices[
        history.bands[-1].span_nodes :
    ]
    if np.max(
        np.linalg.norm(current_actual - current_edge, axis=1),
        initial=0.0,
    ) > attachment_tolerance:
        raise DistributedDoubletError(
            "input newest edge is not attached to the current body cut"
        )

    stage0 = _velocities(history, time, velocity_provider)
    seam0 = _seam_velocity_error(history, stage0)
    if seam0 > seam_velocity_tolerance:
        raise DistributedDoubletError(
            "stage-0 duplicate seam velocities disagree"
        )
    endpoint_edge = _attached_edge(
        history,
        time + dt,
        attached_edge_provider,
    )
    predicted = _update_history(
        history,
        tuple(dt * value for value in stage0),
        endpoint_edge,
    )
    stage1 = _velocities(
        predicted,
        time + dt,
        velocity_provider,
    )
    seam1 = _seam_velocity_error(predicted, stage1)
    if seam1 > seam_velocity_tolerance:
        raise DistributedDoubletError(
            "stage-1 duplicate seam velocities disagree"
        )
    displacement = tuple(
        0.5 * dt * (first + second)
        for first, second in zip(stage0, stage1)
    )
    advanced = _update_history(
        history,
        displacement,
        endpoint_edge,
    )
    report = advanced.continuity_report()
    attachment_error = float(
        np.max(
            np.linalg.norm(
                advanced.bands[-1].surface.vertices[
                    advanced.bands[-1].span_nodes :
                ]
                - endpoint_edge,
                axis=1,
            ),
            initial=0.0,
        )
    )
    strength_residual = max(
        float(
            np.max(
                np.abs(
                    old.surface.face_mu - new.surface.face_mu
                ),
                initial=0.0,
            )
        )
        for old, new in zip(history.bands, advanced.bands)
    )
    area_ratio = _minimum_area_ratio(history, advanced)
    passed = (
        seam0 <= seam_velocity_tolerance
        and seam1 <= seam_velocity_tolerance
        and attachment_error <= attachment_tolerance
        and strength_residual <= strength_tolerance
        and area_ratio >= minimum_face_area_ratio
        and report.compatible
    )
    return ConstrainedWakeAdvectionStep(
        history=advanced,
        report=ConstrainedWakeAdvectionReport(
            dt=float(dt),
            stage0_seam_velocity_error=seam0,
            stage1_seam_velocity_error=seam1,
            attached_edge_error=attachment_error,
            material_strength_residual=strength_residual,
            minimum_face_area_ratio=area_ratio,
            history=report,
            passed=passed,
        ),
    )

