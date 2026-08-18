"""Fail-closed Ptera geometry-history adapter for bounded FluxV v5b smokes.

The adapter deliberately does not run a wake model or own aerodynamic loads.  It
only exposes the already-declared benchmark ``Movement`` geometry in the structured
corner convention consumed by :mod:`claim_runtime.hirato_live_shadow`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pterasoftware as ps

from .baik2012 import BAIK_2012_CASES, build_baik_movement
from .cases import IZRAELEVITZ_2017_FIG14_SCHERER, YANG_2025
from .ptera_adapter import (
    build_izraelevitz_scherer_movement,
    build_yang2025_movement,
)


InitialVelocityMode = Literal["ptera_zero", "periodic_backward"]


@dataclass(frozen=True)
class PteraStructuredHistory:
    """One Ptera wing represented as a chronological structured material grid."""

    corners_history: np.ndarray
    corner_velocity_history: np.ndarray
    delta_time_s: float
    steps_per_cycle: int
    final_cycle_indices: np.ndarray
    u_infinity_gp1_m_s: np.ndarray
    gp1_to_wind_rotation: np.ndarray
    reference_area_m2: float
    rho_kg_m3: float
    freestream_m_s: float
    ptera_symmetric_full_mesh: bool
    hirato_mirror_symmetry: bool
    initial_velocity_mode: str


@dataclass(frozen=True)
class CrossPaperSmokeInput:
    """Declared minimum representative condition and its adapted Ptera history."""

    paper_case_id: str
    condition_id: str
    movement: Any
    manifest: dict[str, Any]
    history: PteraStructuredHistory


def _wing_corners_gp1(wing: Any, *, seam_atol: float) -> np.ndarray:
    panels = wing.panels
    if panels is None or np.asarray(panels).ndim != 2:
        raise ValueError("Ptera wing must expose one 2-D panel array")
    nc, ns = panels.shape
    if nc < 1 or ns < 1:
        raise ValueError("Ptera wing panel array must be non-empty")

    # Ptera panels are chord-major.  Its vertex names are front/back in chord and
    # left/right in span, which maps exactly onto [chord vertex, span vertex].
    corners = np.empty((nc + 1, ns + 1, 3), dtype=float)
    for i in range(nc):
        for j in range(ns):
            panel = panels[i, j]
            fl = np.asarray(panel.Flpp_GP1_CgP1, dtype=float)
            fr = np.asarray(panel.Frpp_GP1_CgP1, dtype=float)
            bl = np.asarray(panel.Blpp_GP1_CgP1, dtype=float)
            br = np.asarray(panel.Brpp_GP1_CgP1, dtype=float)
            if not np.all(np.isfinite(np.stack((fl, fr, bl, br)))):
                raise ValueError("Ptera panel contains non-finite GP1 coordinates")
            if j + 1 < ns:
                right = panels[i, j + 1]
                if not np.allclose(
                    fr, right.Flpp_GP1_CgP1, rtol=0.0, atol=seam_atol
                ) or not np.allclose(br, right.Blpp_GP1_CgP1, rtol=0.0, atol=seam_atol):
                    raise ValueError("Ptera spanwise panel seam is not conformal")
            if i + 1 < nc:
                aft = panels[i + 1, j]
                if not np.allclose(
                    bl, aft.Flpp_GP1_CgP1, rtol=0.0, atol=seam_atol
                ) or not np.allclose(br, aft.Frpp_GP1_CgP1, rtol=0.0, atol=seam_atol):
                    raise ValueError("Ptera chordwise panel seam is not conformal")
            corners[i, j] = fl
            if j == ns - 1:
                corners[i, ns] = fr
            if i == nc - 1:
                corners[nc, j] = bl
                if j == ns - 1:
                    corners[nc, ns] = br
    return corners


def _steps_per_cycle(movement: Any) -> int:
    ratio = float(movement.lcm_period) / float(movement.delta_time)
    rounded = int(round(ratio))
    if rounded < 2 or not np.isclose(ratio, rounded, rtol=0.0, atol=1.0e-9):
        raise ValueError("movement period is not an integer number of time steps")
    return rounded


def _final_complete_cycle_indices(num_steps: int, steps_per_cycle: int) -> np.ndarray:
    if num_steps < steps_per_cycle:
        raise ValueError("history does not contain one complete cycle")
    last = num_steps - 1
    # Ptera may include t=N*T because floating-point ceil turns N*steps into N*steps+1.
    if last > 0 and last % steps_per_cycle == 0:
        end = last
    else:
        end = num_steps
    start = end - steps_per_cycle
    if start < 0:
        raise ValueError("history does not contain a non-duplicated complete cycle")
    return np.arange(start, end, dtype=int)


def structured_history_from_movement(
    movement: Any,
    *,
    initial_velocity_mode: InitialVelocityMode = "ptera_zero",
    airplane_id: int = 0,
    wing_id: int = 0,
    seam_atol: float = 2.0e-12,
) -> PteraStructuredHistory:
    """Extract GP1 corner positions and actual material velocities from Movement.

    ``ptera_zero`` reproduces Ptera's step-zero apparent-motion convention after the
    sign conversion to actual body velocity. ``periodic_backward`` uses the point at
    phase ``1-dt/T`` as the predecessor and is suitable only for a declared periodic
    standalone spin-up.  All later velocities are ordinary backward differences.
    """

    if initial_velocity_mode not in ("ptera_zero", "periodic_backward"):
        raise ValueError("unsupported initial_velocity_mode")
    problem = ps.problems.UnsteadyProblem(movement, only_final_results=False)
    if not problem.steady_problems:
        raise ValueError("Ptera generated an empty unsteady problem")

    corners: list[np.ndarray] = []
    selected_wings: list[Any] = []
    for steady in problem.steady_problems:
        if airplane_id < 0 or airplane_id >= len(steady.airplanes):
            raise ValueError("airplane_id is outside Ptera history")
        airplane = steady.airplanes[airplane_id]
        if wing_id < 0 or wing_id >= len(airplane.wings):
            raise ValueError("wing_id is outside Ptera airplane")
        wing = airplane.wings[wing_id]
        selected_wings.append(wing)
        corners.append(_wing_corners_gp1(wing, seam_atol=seam_atol))
    corners_history = np.stack(corners)
    if not np.all(np.isfinite(corners_history)):
        raise ValueError("structured corner history is non-finite")

    delta_time = float(movement.delta_time)
    steps_per_cycle = _steps_per_cycle(movement)
    velocities = np.empty_like(corners_history)
    velocities[1:] = np.diff(corners_history, axis=0) / delta_time
    if initial_velocity_mode == "ptera_zero":
        velocities[0] = 0.0
    else:
        predecessor = steps_per_cycle - 1
        if predecessor >= len(corners_history):
            raise ValueError("periodic predecessor is absent from Ptera history")
        if len(corners_history) > steps_per_cycle and not np.allclose(
            corners_history[0],
            corners_history[steps_per_cycle],
            rtol=0.0,
            atol=5.0e-11,
        ):
            raise ValueError("Ptera history does not close at the declared period")
        velocities[0] = (corners_history[0] - corners_history[predecessor]) / delta_time

    first_problem = problem.steady_problems[0]
    first_airplane = first_problem.airplanes[airplane_id]
    first_wing = selected_wings[0]
    op = first_problem.operating_point
    rotation = np.asarray(op.T_pas_GP1_CgP1_to_W_CgP1, dtype=float)[:3, :3]
    reference_area = first_airplane.s_ref
    if reference_area is None or float(reference_area) <= 0.0:
        raise ValueError("Ptera airplane has no positive reference area")

    # Ptera type-4 symmetric wings are explicitly meshed on both sides.  Passing an
    # additional y=0 image to Hirato would double count bound and wake induction.
    is_full_symmetric_mesh = bool(
        first_wing.symmetric and int(first_wing.symmetry_type) == 4
    )
    return PteraStructuredHistory(
        corners_history=corners_history,
        corner_velocity_history=velocities,
        delta_time_s=delta_time,
        steps_per_cycle=steps_per_cycle,
        final_cycle_indices=_final_complete_cycle_indices(
            len(corners_history), steps_per_cycle
        ),
        u_infinity_gp1_m_s=np.asarray(op.vInf_GP1__E, dtype=float).copy(),
        gp1_to_wind_rotation=rotation.copy(),
        reference_area_m2=float(reference_area),
        rho_kg_m3=float(op.rho),
        freestream_m_s=float(op.vCg__E),
        ptera_symmetric_full_mesh=is_full_symmetric_mesh,
        hirato_mirror_symmetry=False,
        initial_velocity_mode=initial_velocity_mode,
    )


def coefficients_from_gp1_force(
    force_gp1_n: np.ndarray,
    history: PteraStructuredHistory,
) -> dict[str, np.ndarray]:
    """Apply the existing benchmark wind-axis signs and full-area normalization."""

    force = np.asarray(force_gp1_n, dtype=float)
    if force.shape[-1:] != (3,) or not np.all(np.isfinite(force)):
        raise ValueError("force_gp1_n must be finite with final dimension three")
    force_w = np.einsum("ij,...j->...i", history.gp1_to_wind_rotation, force)
    q_area = (
        0.5
        * history.rho_kg_m3
        * history.freestream_m_s**2
        * history.reference_area_m2
    )
    thrust = force_w[..., 0]
    lift = -force_w[..., 2]
    return {
        "force_w_n": force_w,
        "lift_n": lift,
        "thrust_n": thrust,
        "drag_n": -thrust,
        "CL": lift / q_area,
        "CT": thrust / q_area,
        "CD": -thrust / q_area,
    }


def build_crosspaper_smoke_input(
    paper_case_id: Literal["yang2025", "izraelevitz2017_fig14", "baik2012"],
    *,
    initial_velocity_mode: InitialVelocityMode = "ptera_zero",
) -> CrossPaperSmokeInput:
    """Build one frozen, minimum representative condition per benchmark paper."""

    if paper_case_id == "yang2025":
        condition_id = "aoa15_deg"
        movement, manifest = build_yang2025_movement(15.0, quality="smoke")
    elif paper_case_id == "izraelevitz2017_fig14":
        # Strong positive experimental thrust near the centre of the digitized phase
        # sweep, without selecting an endpoint condition.
        condition_id = "theta15_deg_psi60_deg"
        movement, manifest = build_izraelevitz_scherer_movement(
            15.0, 60.0, quality="smoke"
        )
    elif paper_case_id == "baik2012":
        # W2 combines the high reduced frequency and large pitch family; W4 remains a
        # subsequent large-heave stress case rather than the minimum smoke.
        condition_id = "W2"
        movement, manifest = build_baik_movement(BAIK_2012_CASES["W2"], quality="smoke")
    else:
        raise ValueError(f"unsupported cross-paper smoke: {paper_case_id}")

    history = structured_history_from_movement(
        movement,
        initial_velocity_mode=initial_velocity_mode,
    )
    expected = {
        "yang2025": YANG_2025.area_m2,
        "izraelevitz2017_fig14": IZRAELEVITZ_2017_FIG14_SCHERER.area_m2,
        "baik2012": BAIK_2012_CASES["W2"].area_m2,
    }[paper_case_id]
    if not np.isclose(history.reference_area_m2, expected, rtol=0.0, atol=1.0e-15):
        raise ValueError("Ptera reference area differs from benchmark declaration")
    return CrossPaperSmokeInput(
        paper_case_id=paper_case_id,
        condition_id=condition_id,
        movement=movement,
        manifest=manifest,
        history=history,
    )
