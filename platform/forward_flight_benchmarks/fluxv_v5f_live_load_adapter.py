"""Read-only adapter from a live Ptera load state to the v5f load kernel.

Ptera's :meth:`_calculate_loads` keeps the four LineVortex-center velocities as
local variables.  This module reconstructs those *operands*, in the same order
and with the same Ptera methods, without running ``process_solver_loads`` and
without writing forces, moments, bound circulation, or wake state.

The adapter deliberately calculates only a load *delta*.  It does not attach
the delta to a solver.  A future time-marching integration must separately
prove that it applies the returned arrays exactly once.

The material-release call contract is explicit:

* ``surface_release_current = where(active_current, gamma_lev_current, 0)``;
* for steps after zero, ``wake_release_previous = gamma_lev_previous``;
* at step zero, prior material-LEV state must be absent (``None``), and both
  prior gamma and prior wake release are represented by exact positive zeros.

These identities are checked immediately before calling
:func:`calculate_native_surface_release_load_correction`.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final, Mapping

import numpy as np

from fluxvortex.solver import UVPMHybridSolver

from .fluxv_v5f_native_load import (
    LEG_NAMES,
    calculate_native_surface_release_load_correction,
)


_CENTER_ATTRIBUTES: Final[dict[str, str]] = {
    "right": "stackCblvpr_GP1_CgP1",
    "front": "stackCblvpf_GP1_CgP1",
    "left": "stackCblvpl_GP1_CgP1",
    "back": "stackCblvpb_GP1_CgP1",
}
_VECTOR_ATTRIBUTES: Final[dict[str, str]] = {
    "right": "stackRbrv_GP1",
    "front": "stackFbrv_GP1",
    "left": "stackLbrv_GP1",
    "back": "stackBbrv_GP1",
}
_MOVEMENT_METHODS: Final[dict[str, str]] = {
    "right": "_calculate_current_movement_velocities_at_right_leg_centers",
    "front": "_calculate_current_movement_velocities_at_front_leg_centers",
    "left": "_calculate_current_movement_velocities_at_left_leg_centers",
    "back": "_calculate_current_movement_velocities_at_back_leg_centers",
}


def _readonly_copy(value: Any, *, dtype: Any = float) -> np.ndarray:
    array = np.asarray(value, dtype=dtype).copy()
    array.setflags(write=False)
    return array


def _readonly_mapping(
    values: Mapping[str, np.ndarray], *, dtype: Any = float
) -> Mapping[str, np.ndarray]:
    return MappingProxyType(
        {name: _readonly_copy(values[name], dtype=dtype) for name in values}
    )


@dataclass(frozen=True)
class PteraNativeLiveLoadOperands:
    """Immutable copies of every array needed by the v5f surface-load kernel."""

    current_step: int
    rho_kg_m3: float
    delta_time_s: float
    topology: Mapping[str, np.ndarray]
    panel_area_m2: np.ndarray
    panel_normal_gp1: np.ndarray
    panel_collocation_gp1_cgp1_m: np.ndarray
    leg_vectors_gp1_m: Mapping[str, np.ndarray]
    leg_centers_gp1_cgp1_m: Mapping[str, np.ndarray]
    leg_solution_velocities_gp1_m_s: Mapping[str, np.ndarray]
    leg_movement_velocities_gp1_m_s: Mapping[str, np.ndarray]
    leg_velocities_gp1_m_s: Mapping[str, np.ndarray]
    bound_singularity_counts: np.ndarray
    wake_singularity_counts: np.ndarray

    @property
    def panel_count(self) -> int:
        return int(self.panel_area_m2.size)


@dataclass(frozen=True)
class PteraNativeLiveLoadDelta:
    """A read-only live-operand capture and its uncommitted correction."""

    operands: PteraNativeLiveLoadOperands
    gamma_lev_current_m2_s: np.ndarray
    gamma_lev_previous_m2_s: np.ndarray
    active_current: np.ndarray
    surface_release_current_m2_s: np.ndarray
    wake_release_previous_m2_s: np.ndarray
    correction: Mapping[str, Any]
    call_contract: Mapping[str, str]


def _build_panel_topology(solver: UVPMHybridSolver) -> Mapping[str, np.ndarray]:
    airplane_indices: list[int] = []
    wing_indices: list[int] = []
    chordwise_indices: list[int] = []
    spanwise_indices: list[int] = []
    for airplane_index, airplane in enumerate(solver.current_airplanes):
        for wing_index, wing in enumerate(airplane.wings):
            panels = wing.panels
            if panels is None:
                raise RuntimeError("live Ptera Wing has no Panel grid")
            for panel in np.ravel(panels):
                chordwise = panel.local_chordwise_position
                spanwise = panel.local_spanwise_position
                if chordwise is None or spanwise is None:
                    raise RuntimeError("live Ptera Panel topology is incomplete")
                airplane_indices.append(airplane_index)
                wing_indices.append(wing_index)
                chordwise_indices.append(int(chordwise))
                spanwise_indices.append(int(spanwise))
    topology = {
        "airplane_index": np.asarray(airplane_indices, dtype=np.int64),
        "wing_index": np.asarray(wing_indices, dtype=np.int64),
        "chordwise_index": np.asarray(chordwise_indices, dtype=np.int64),
        "spanwise_index": np.asarray(spanwise_indices, dtype=np.int64),
    }
    if len(airplane_indices) != int(solver.num_panels):
        raise RuntimeError("live Ptera panel order does not match solver.num_panels")
    return _readonly_mapping(topology, dtype=np.int64)


def _require_vector_source(
    solver: UVPMHybridSolver, attribute: str, *, panel_count: int
) -> np.ndarray:
    array = np.asarray(getattr(solver, attribute), dtype=float)
    if array.shape != (panel_count, 3) or not np.all(np.isfinite(array)):
        raise RuntimeError(
            f"live Ptera source {attribute!r} must be finite with shape "
            "(num_panels, 3)"
        )
    return array


def _require_solved_parent_load_state(
    solver: UVPMHybridSolver, *, panel_count: int
) -> None:
    """Fail closed unless the current bound state and parent loads are committed."""

    gamma = np.asarray(
        getattr(solver, "_current_bound_vortex_strengths", None), dtype=float
    )
    if gamma.shape != (panel_count,) or not np.all(np.isfinite(gamma)):
        raise RuntimeError("current Ptera bound circulation has not been solved")

    traversed = 0
    for airplane in solver.current_airplanes:
        for wing in airplane.wings:
            panels = wing.panels
            if panels is None:
                raise RuntimeError("live Ptera Wing has no Panel grid")
            for panel in np.ravel(panels):
                ring = panel.ring_vortex
                if ring is None or float(ring.strength) != float(gamma[traversed]):
                    raise RuntimeError(
                        "current Ptera Panel rings disagree with solved circulation"
                    )
                for attribute in ("forces_GP1", "moments_GP1_CgP1"):
                    value = getattr(panel, attribute, None)
                    array = np.asarray(value, dtype=float)
                    if array.shape != (3,) or not np.all(np.isfinite(array)):
                        raise RuntimeError(
                            "capture requires the current parent load calculation "
                            "to have completed"
                        )
                traversed += 1
    if traversed != panel_count:
        raise RuntimeError("live Ptera panel traversal disagrees with panel count")

    aic = np.asarray(
        getattr(solver, "_currentGridWingWingInfluences__E", None), dtype=float
    )
    wake = np.asarray(
        getattr(solver, "_currentStackWakeWingInfluences__E", None), dtype=float
    )
    freestream = np.asarray(
        getattr(solver, "_currentStackFreestreamWingInfluences__E", None), dtype=float
    )
    if (
        aic.shape != (panel_count, panel_count)
        or wake.shape != (panel_count,)
        or freestream.shape != (panel_count,)
        or not np.all(np.isfinite(aic))
        or not np.all(np.isfinite(wake))
        or not np.all(np.isfinite(freestream))
    ):
        raise RuntimeError("current Ptera AIC/RHS state is incomplete")


def capture_ptera_native_load_operands(
    solver: UVPMHybridSolver,
) -> PteraNativeLiveLoadOperands:
    """Capture the exact read-only operands of Ptera's current load formula.

    The four velocity evaluations intentionally share one pair of zero-initialized
    singularity counters, matching Ptera's parent ``_calculate_loads`` method.
    No solver-owned array is returned by reference.
    """

    if not isinstance(solver, UVPMHybridSolver):
        raise TypeError("solver must be a UVPMHybridSolver")
    if not hasattr(solver, "_current_step"):
        raise RuntimeError("solver has no initialized current Ptera step")
    step = int(solver._current_step)
    if step < 0:
        raise RuntimeError("solver current step must be nonnegative")
    panel_count = int(solver.num_panels)
    if panel_count <= 0:
        raise RuntimeError("solver must have at least one live Panel")

    current_wake_strengths = np.asarray(
        getattr(solver, "_current_wake_vortex_strengths", None), dtype=float
    )
    if current_wake_strengths.ndim != 1 or not np.all(
        np.isfinite(current_wake_strengths)
    ):
        raise RuntimeError("current Ptera wake-strength state is invalid")
    if step == 0:
        empty_wake_arrays = (
            "_current_wake_vortex_strengths",
            "_current_wake_vortex_ages",
            "_currentStackWakeRc0s",
            "_currentStackBrwrvp_GP1_CgP1",
            "_currentStackFrwrvp_GP1_CgP1",
            "_currentStackFlwrvp_GP1_CgP1",
            "_currentStackBlwrvp_GP1_CgP1",
        )
        for attribute in empty_wake_arrays:
            wake_array = np.asarray(getattr(solver, attribute, None), dtype=float)
            if wake_array.size != 0 or not np.all(np.isfinite(wake_array)):
                raise RuntimeError("Ptera step zero must not contain a prior wake row")
    _require_solved_parent_load_state(solver, panel_count=panel_count)

    rho = float(solver.current_operating_point.rho)
    delta_time = float(solver.delta_time)
    if not np.isfinite(rho) or rho <= 0.0:
        raise RuntimeError("live Ptera density must be finite and positive")
    if not np.isfinite(delta_time) or delta_time <= 0.0:
        raise RuntimeError("live Ptera time step must be finite and positive")

    centers: dict[str, np.ndarray] = {}
    vectors: dict[str, np.ndarray] = {}
    solution_velocities: dict[str, np.ndarray] = {}
    movement_velocities: dict[str, np.ndarray] = {}
    velocities: dict[str, np.ndarray] = {}
    bound_singularity_counts = np.zeros(4, dtype=np.int64)
    wake_singularity_counts = np.zeros(4, dtype=np.int64)
    for leg in LEG_NAMES:
        center = _require_vector_source(
            solver, _CENTER_ATTRIBUTES[leg], panel_count=panel_count
        )
        vector = _require_vector_source(
            solver, _VECTOR_ATTRIBUTES[leg], panel_count=panel_count
        )
        solution_velocity = np.asarray(
            solver.calculate_solution_velocity(
                stackP_GP1_CgP1=center,
                bound_singularity_counts=bound_singularity_counts,
                wake_singularity_counts=wake_singularity_counts,
            ),
            dtype=float,
        )
        movement_method = getattr(solver, _MOVEMENT_METHODS[leg])
        movement_velocity = np.asarray(movement_method(), dtype=float)
        if (
            solution_velocity.shape != (panel_count, 3)
            or movement_velocity.shape != (panel_count, 3)
            or not np.all(np.isfinite(solution_velocity))
            or not np.all(np.isfinite(movement_velocity))
        ):
            raise RuntimeError(f"live Ptera {leg} velocity operands are invalid")
        total_velocity = solution_velocity + movement_velocity
        if not np.all(np.isfinite(total_velocity)):
            raise RuntimeError(f"live Ptera {leg} total velocity is non-finite")
        centers[leg] = center
        vectors[leg] = vector
        solution_velocities[leg] = solution_velocity
        movement_velocities[leg] = movement_velocity
        velocities[leg] = total_velocity

    area = np.asarray(solver.panel_areas, dtype=float)
    normal = _require_vector_source(
        solver, "stackUnitNormals_GP1", panel_count=panel_count
    )
    collocation = _require_vector_source(
        solver, "stackCpp_GP1_CgP1", panel_count=panel_count
    )
    if area.shape != (panel_count,) or not np.all(np.isfinite(area)):
        raise RuntimeError("live Ptera panel areas are invalid")
    if np.any(area <= 0.0):
        raise RuntimeError("live Ptera panel areas must be positive")

    return PteraNativeLiveLoadOperands(
        current_step=step,
        rho_kg_m3=rho,
        delta_time_s=delta_time,
        topology=_build_panel_topology(solver),
        panel_area_m2=_readonly_copy(area),
        panel_normal_gp1=_readonly_copy(normal),
        panel_collocation_gp1_cgp1_m=_readonly_copy(collocation),
        leg_vectors_gp1_m=_readonly_mapping(vectors),
        leg_centers_gp1_cgp1_m=_readonly_mapping(centers),
        leg_solution_velocities_gp1_m_s=_readonly_mapping(solution_velocities),
        leg_movement_velocities_gp1_m_s=_readonly_mapping(movement_velocities),
        leg_velocities_gp1_m_s=_readonly_mapping(velocities),
        bound_singularity_counts=_readonly_copy(
            bound_singularity_counts, dtype=np.int64
        ),
        wake_singularity_counts=_readonly_copy(wake_singularity_counts, dtype=np.int64),
    )


def _panel_scalar_state(
    value: np.ndarray, *, name: str, panel_count: int
) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (panel_count,):
        raise ValueError(f"{name} must have shape (num_panels,)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array.copy()


def _panel_active_state(value: bool | np.ndarray, *, panel_count: int) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape == ():
        if not np.issubdtype(raw.dtype, np.bool_):
            raise TypeError("active_current must be boolean")
        return np.full(panel_count, bool(raw), dtype=bool)
    if raw.shape != (panel_count,) or not np.issubdtype(raw.dtype, np.bool_):
        raise TypeError(
            "active_current must be a boolean scalar or boolean num_panels array"
        )
    return raw.astype(bool, copy=True)


def calculate_native_load_delta_from_live_operands(
    operands: PteraNativeLiveLoadOperands,
    *,
    gamma_lev_current_m2_s: np.ndarray,
    gamma_lev_previous_m2_s: np.ndarray | None,
    active_current: bool | np.ndarray,
) -> PteraNativeLiveLoadDelta:
    """Call the pure v5f correction kernel with an enforced release ledger."""

    if not isinstance(operands, PteraNativeLiveLoadOperands):
        raise TypeError("operands must be PteraNativeLiveLoadOperands")
    current = _panel_scalar_state(
        gamma_lev_current_m2_s,
        name="gamma_lev_current_m2_s",
        panel_count=operands.panel_count,
    )
    active = _panel_active_state(active_current, panel_count=operands.panel_count)
    surface_current = np.where(active, current, 0.0)

    if operands.current_step == 0:
        if gamma_lev_previous_m2_s is not None:
            raise ValueError(
                "gamma_lev_previous_m2_s must be None at step zero because no "
                "prior material LEV or corresponding wake release exists"
            )
        previous = np.zeros(operands.panel_count, dtype=float)
        wake_previous = np.zeros(operands.panel_count, dtype=float)
    else:
        if gamma_lev_previous_m2_s is None:
            raise ValueError("gamma_lev_previous_m2_s is required after step zero")
        previous = _panel_scalar_state(
            gamma_lev_previous_m2_s,
            name="gamma_lev_previous_m2_s",
            panel_count=operands.panel_count,
        )
        wake_previous = previous.copy()

    # Hard calling-side identities.  Keep these checks adjacent to the native
    # kernel invocation so later time-marching work cannot silently conflate the
    # current surface release with the prior wake release.
    expected_surface = np.where(active, current, 0.0)
    if not np.array_equal(surface_current, expected_surface):
        raise AssertionError("surface-release current-state contract failed")
    if operands.current_step > 0 and not np.array_equal(wake_previous, previous):
        raise AssertionError("wake-release previous-state contract failed")
    if operands.current_step == 0 and (
        np.any(previous != 0.0) or np.any(wake_previous != 0.0)
    ):
        raise AssertionError("step-zero prior-state absence contract failed")

    correction = calculate_native_surface_release_load_correction(
        rho_kg_m3=operands.rho_kg_m3,
        delta_time_s=operands.delta_time_s,
        current_step=operands.current_step,
        topology=operands.topology,
        surface_release_current_m2_s=surface_current,
        wake_release_previous_m2_s=wake_previous,
        gamma_lev_current_m2_s=current,
        gamma_lev_previous_m2_s=previous,
        active_current=active,
        panel_area_m2=operands.panel_area_m2,
        panel_normal_gp1=operands.panel_normal_gp1,
        panel_collocation_gp1_cgp1_m=operands.panel_collocation_gp1_cgp1_m,
        leg_vectors_gp1_m=operands.leg_vectors_gp1_m,
        leg_velocities_gp1_m_s=operands.leg_velocities_gp1_m_s,
        leg_centers_gp1_cgp1_m=operands.leg_centers_gp1_cgp1_m,
    )
    contract = MappingProxyType(
        {
            "surface_release_current": ("where(active_current, gamma_lev_current, 0)"),
            "wake_release_previous": ("zero_at_step_0_else_gamma_lev_previous"),
            "commit": "none_delta_only",
        }
    )
    return PteraNativeLiveLoadDelta(
        operands=operands,
        gamma_lev_current_m2_s=_readonly_copy(current),
        gamma_lev_previous_m2_s=_readonly_copy(previous),
        active_current=_readonly_copy(active, dtype=bool),
        surface_release_current_m2_s=_readonly_copy(surface_current),
        wake_release_previous_m2_s=_readonly_copy(wake_previous),
        correction=correction,
        call_contract=contract,
    )


def calculate_live_native_surface_release_load_delta(
    solver: UVPMHybridSolver,
    *,
    gamma_lev_current_m2_s: np.ndarray,
    gamma_lev_previous_m2_s: np.ndarray | None,
    active_current: bool | np.ndarray,
) -> PteraNativeLiveLoadDelta:
    """Capture one live Ptera state and return its uncommitted v5f load delta."""

    operands = capture_ptera_native_load_operands(solver)
    return calculate_native_load_delta_from_live_operands(
        operands,
        gamma_lev_current_m2_s=gamma_lev_current_m2_s,
        gamma_lev_previous_m2_s=gamma_lev_previous_m2_s,
        active_current=active_current,
    )


__all__ = [
    "PteraNativeLiveLoadDelta",
    "PteraNativeLiveLoadOperands",
    "calculate_live_native_surface_release_load_delta",
    "calculate_native_load_delta_from_live_operands",
    "capture_ptera_native_load_operands",
]
