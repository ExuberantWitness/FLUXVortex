"""Mechanical time marcher for the FluxV v5f native material-LEV candidate.

This module is intentionally narrower than a production time marcher.  It
keeps :class:`fluxvortex.solver.UVPMHybridSolver` as the owner of Ptera's
bound AIC, prescribed TE wake, and Kutta--Joukowski plus ``dGamma/dt`` load
ledger.  For every staged time step it can:

* observe Hirato Eq. 6 on the live Ptera lattice;
* solve the literal ``[A B; H 0]`` system with a nascent material LEV and
  pseudovortex evaluated by Hirato Eq. 25;
* add the source-derived surface-release load correction to Ptera's load;
* patch the next TE row to ``Gamma_b + Gamma_LEV``; and
* atomically publish and convect the material rings.

This remains mechanical evidence only.  Core/time refinement and physical
boundedness must pass before any paper prediction is allowed.  No experiment
or target load is read here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Sequence

import numpy as np
from pterasoftware import _aerodynamics_functions, _functions, _transformations

from fluxvortex.solver import UVPMHybridSolver

from claim_runtime.hirato_equations import (
    RAMESH_2014_LDVM_V25_BIRTH_SOURCE_TAG,
    embed_chord_normal_displacement,
    first_lev_displacement_ramesh_2d,
    first_lev_displacement_ramesh_2014_ldvm_v25,
)
from claim_runtime.hirato_shadow import ring_field_velocity_lamb_oseen

from .fluxv_v5f_live_load_adapter import (
    PteraNativeLiveLoadDelta,
    calculate_live_native_surface_release_load_delta,
)
from .fluxv_v5f_material_state import (
    PteraMaterialLEVProposal,
    PteraMaterialLEVSnapshot,
    PteraMaterialLEVView,
    PteraNativeMaterialLEVState,
    ptera_to_shadow_forward_rings,
)
from .fluxv_v5f_native_lev import (
    AUGMENTED_RESIDUAL_ATOL,
    PteraNativeAugmentedSolution,
    PteraNativeLESPObserver,
    build_ptera_native_lesp_observer,
    solve_ptera_native_augmented_system,
)


CONTINUATION_STATUS: Final = "mechanical_continuation_unscored"
FORCE_SCORING_STATUS: Final = "blocked_mechanical_time_march_unscored"
REGISTERED_CORE_RADIUS_RATIOS: Final = (0.10, 0.25, 0.49)
PTERA_LDVM_V25_BIRTH_MAPPING_STATUS: Final = (
    "ramesh-v2.5-geometry-law_mapped-to-ptera-native-time-layer;"
    "current-step-provisional-TEV-unavailable"
)


@dataclass(frozen=True)
class NativeMaterialLEVTimeMarchConfig:
    """Frozen controls for the unscored v5f mechanical time marcher.

    ``core_radius_ratio`` is a preregistered mechanical sensitivity, not an
    accuracy parameter.  It multiplies one configuration-wide length scale:
    the smaller of the global minimum strip width and the Ramesh first-birth
    displacement evaluated at ``lesp_critical``.  Every newborn ring in that
    discretization therefore receives the same core.  The core remains fixed
    for the material ring; no age, viscosity, or Squire-growth law is applied.
    """

    enabled: bool = False
    lesp_critical: float = np.inf
    core_radius_ratio: float = 0.25

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, (bool, np.bool_)):
            raise TypeError("enabled must be boolean")
        threshold = float(self.lesp_critical)
        if np.isnan(threshold) or threshold <= 0.0:
            raise ValueError("lesp_critical must be positive and not NaN")
        ratio = float(self.core_radius_ratio)
        if ratio not in REGISTERED_CORE_RADIUS_RATIOS:
            raise ValueError(
                "core_radius_ratio must be one of the preregistered values "
                f"{REGISTERED_CORE_RADIUS_RATIOS}"
            )


DEFAULT_NATIVE_TIME_MARCH_CONFIG: Final = NativeMaterialLEVTimeMarchConfig()


@dataclass(frozen=True)
class NativeMaterialLEVStepReport:
    """Immutable evidence ledger for one fully committed vertical-slice step."""

    step: int
    active: np.ndarray
    q_by_strip_m2_s: np.ndarray
    a0_pre: np.ndarray
    a0_post: np.ndarray
    core_radius_by_strip_m: np.ndarray
    augmented_solution: PteraNativeAugmentedSolution
    load_delta: PteraNativeLiveLoadDelta | None
    material_snapshot_before: PteraMaterialLEVSnapshot
    created_ring_ids: np.ndarray
    new_sheet: np.ndarray
    parent_wake_front_strengths_m2_s: np.ndarray
    rear_bound_strengths_m2_s: np.ndarray
    next_wake_front_strengths_m2_s: np.ndarray
    material_snapshot: PteraMaterialLEVSnapshot
    eq9_max_abs_m2_s: float
    force_scoring_status: str
    continuation_status: str


@dataclass(frozen=True)
class PteraMappedLDVMV25BirthVelocityLedger:
    """Causal local-edge velocity contract for the isolated v5f2 birth law.

    Every vector has shape ``(strip, edge_endpoint, GP1_xyz)`` and is observed
    from Ptera's Earth frame at the current staged time ``t_n``.  The apparent
    movement term is ``-(x_LE(t_n)-x_LE(t_{n-1}))/dt`` and is exact zero at
    step zero, matching Ptera's own movement convention.  Because the source
    ``q_LE`` is a two-dimensional section velocity, the summed 3-D transfer
    velocity is explicitly projected onto each strip's chord-normal plane;
    ``spanwise_removed_m_s`` records the discarded component.

    The Ptera time layer has the already committed parent TE-origin wake but
    no explicit provisional TEV for the current step.  Consequently this is a
    documented transfer of the Ramesh-v2.5 *geometry law*, not full parity
    with the Fortran ``q_LE`` ledger.  Bound rings, the current newborn LEV,
    and its pseudovortex are deliberately excluded and recorded as zeros.
    """

    step: int
    new_sheet: np.ndarray
    freestream_m_s: np.ndarray
    current_parent_te_wake_induced_m_s: np.ndarray
    committed_old_material_induced_m_s: np.ndarray
    apparent_movement_m_s: np.ndarray
    raw_total_local_edge_velocity_m_s: np.ndarray
    spanwise_removed_m_s: np.ndarray
    total_local_edge_velocity_m_s: np.ndarray
    excluded_bound_induced_m_s: np.ndarray
    excluded_current_newborn_induced_m_s: np.ndarray
    excluded_pseudovortex_induced_m_s: np.ndarray
    source_tag: str
    mapping_status: str


@dataclass
class _PendingMaterialStep:
    observer: PteraNativeLESPObserver
    proposal: PteraMaterialLEVProposal
    active: np.ndarray
    q_by_strip_m2_s: np.ndarray
    a0_pre: np.ndarray
    a0_post: np.ndarray
    core_radius_by_strip_m: np.ndarray
    augmented_solution: PteraNativeAugmentedSolution
    staged_view: PteraMaterialLEVView
    pseudovortex_rings_gp1_m: np.ndarray
    parent_gamma_bound: np.ndarray
    material_snapshot_before: PteraMaterialLEVSnapshot
    created_ring_ids: np.ndarray
    new_sheet: np.ndarray
    parent_wake_front_strengths_m2_s: np.ndarray
    load_delta: PteraNativeLiveLoadDelta | None = None


def _finite_array(value: Any, *, name: str, ndim: int) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def assemble_ptera_mapped_ldvm_v25_local_edge_velocity(
    *,
    step: int,
    new_sheet: np.ndarray | Sequence[bool],
    freestream_m_s: np.ndarray,
    current_parent_te_wake_induced_m_s: np.ndarray,
    committed_old_material_induced_m_s: np.ndarray,
    apparent_movement_m_s: np.ndarray,
    span_tangent_gp1: np.ndarray,
) -> PteraMappedLDVMV25BirthVelocityLedger:
    """Assemble the one-factor v5f2 ``q_LE`` transfer ledger.

    No excluded term is accepted as an input: this prevents a caller from
    silently adding bound, current-newborn, or pseudovortex induction to the
    first/restart placement velocity.
    """

    if isinstance(step, (bool, np.bool_)) or not isinstance(step, (int, np.integer)):
        raise TypeError("step must be a non-negative integer")
    step_value = int(step)
    if step_value < 0:
        raise ValueError("step must be a non-negative integer")
    components = {
        "freestream_m_s": _finite_array(
            freestream_m_s,
            name="freestream_m_s",
            ndim=3,
        ),
        "current_parent_te_wake_induced_m_s": _finite_array(
            current_parent_te_wake_induced_m_s,
            name="current_parent_te_wake_induced_m_s",
            ndim=3,
        ),
        "committed_old_material_induced_m_s": _finite_array(
            committed_old_material_induced_m_s,
            name="committed_old_material_induced_m_s",
            ndim=3,
        ),
        "apparent_movement_m_s": _finite_array(
            apparent_movement_m_s,
            name="apparent_movement_m_s",
            ndim=3,
        ),
    }
    shape = components["freestream_m_s"].shape
    if len(shape) != 3 or shape[1:] != (2, 3):
        raise ValueError("local-edge components must have shape (strip,2,3)")
    if any(component.shape != shape for component in components.values()):
        raise ValueError("all local-edge velocity components must have equal shape")
    starts = np.asarray(new_sheet, dtype=bool)
    if starts.shape != (shape[0],):
        raise ValueError("new_sheet must have one value per local-edge strip")
    span_tangent = _finite_array(
        span_tangent_gp1,
        name="span_tangent_gp1",
        ndim=2,
    )
    if span_tangent.shape != (shape[0], 3):
        raise ValueError("span_tangent_gp1 must have shape (strip,3)")
    span_norm = np.linalg.norm(span_tangent, axis=1)
    if np.any(np.abs(span_norm - 1.0) > 1.0e-10):
        raise ValueError("span_tangent_gp1 must contain unit vectors")
    raw_total = sum(
        (component for component in components.values()),
        np.zeros(shape),
    )
    spanwise = (
        np.einsum("sej,sj->se", raw_total, span_tangent)[:, :, None]
        * span_tangent[:, None, :]
    )
    total = raw_total - spanwise
    if not np.all(np.isfinite(total)):
        raise FloatingPointError("assembled local-edge velocity is non-finite")
    excluded = np.zeros(shape, dtype=float)
    return PteraMappedLDVMV25BirthVelocityLedger(
        step=step_value,
        new_sheet=starts.copy(),
        freestream_m_s=components["freestream_m_s"].copy(),
        current_parent_te_wake_induced_m_s=components[
            "current_parent_te_wake_induced_m_s"
        ].copy(),
        committed_old_material_induced_m_s=components[
            "committed_old_material_induced_m_s"
        ].copy(),
        apparent_movement_m_s=components["apparent_movement_m_s"].copy(),
        raw_total_local_edge_velocity_m_s=raw_total,
        spanwise_removed_m_s=spanwise,
        total_local_edge_velocity_m_s=total,
        excluded_bound_induced_m_s=excluded.copy(),
        excluded_current_newborn_induced_m_s=excluded.copy(),
        excluded_pseudovortex_induced_m_s=excluded.copy(),
        source_tag=RAMESH_2014_LDVM_V25_BIRTH_SOURCE_TAG,
        mapping_status=PTERA_LDVM_V25_BIRTH_MAPPING_STATUS,
    )


def eq25_velocity_from_ptera_material_rings(
    points_gp1_m: np.ndarray | Sequence[Sequence[float]],
    rings_gp1_m: np.ndarray,
    gamma_m2_s: np.ndarray | Sequence[float],
    core_radius_m: np.ndarray | Sequence[float],
    *,
    surface_reflection_transform: np.ndarray | None = None,
) -> np.ndarray:
    """Evaluate Ptera-readable material rings with Hirato Eq. 25.

    Ptera's positive circulation traverses its readable ring in reverse, so
    every ring is converted back to the forward shadow traversal before the
    Eq. 25 finite-segment evaluator is called.  A distinct frozen core is
    accepted for each material ring.  The empty field returns exact positive
    zero and performs no kernel call.
    """

    points = _finite_array(points_gp1_m, name="points_gp1_m", ndim=2)
    rings = _finite_array(rings_gp1_m, name="rings_gp1_m", ndim=3)
    gamma = _finite_array(gamma_m2_s, name="gamma_m2_s", ndim=1)
    core = _finite_array(core_radius_m, name="core_radius_m", ndim=1)
    if points.shape[1:] != (3,) or rings.shape[1:] != (4, 3):
        raise ValueError("points and rings must have shapes (N,3) and (M,4,3)")
    if gamma.shape != (rings.shape[0],) or core.shape != (rings.shape[0],):
        raise ValueError("gamma_m2_s and core_radius_m need one value per ring")
    if np.any(core <= 0.0):
        raise ValueError("every Eq. 25 core radius must be positive")
    if rings.shape[0] == 0:
        return np.zeros_like(points)

    forward = ptera_to_shadow_forward_rings(rings)

    def direct(evaluation_points: np.ndarray) -> np.ndarray:
        velocity = np.zeros_like(evaluation_points)
        for ring, strength, radius in zip(forward, gamma, core, strict=True):
            velocity += ring_field_velocity_lamb_oseen(
                evaluation_points,
                ring[None, :, :],
                np.asarray([strength], dtype=float),
                float(radius),
            )
        return velocity

    result = direct(points)
    if surface_reflection_transform is not None:
        reflection = _finite_array(
            surface_reflection_transform,
            name="surface_reflection_transform",
            ndim=2,
        )
        if reflection.shape != (4, 4):
            raise ValueError("surface_reflection_transform must have shape (4,4)")
        reflected_points = _transformations.apply_T_to_vectors(
            reflection,
            points,
            has_point=True,
        )
        image_velocity = direct(reflected_points)
        result = result + _transformations.apply_T_to_vectors(
            reflection,
            image_velocity,
            has_point=False,
        )
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("Eq. 25 material velocity is non-finite")
    return result


def _count_strips(solver: UVPMHybridSolver) -> int:
    count = 0
    first_problem = solver.steady_problems[0]
    for airplane in first_problem.airplanes:
        for wing in airplane.wings:
            panels = np.asarray(wing.panels, dtype=object)
            if panels.ndim != 2 or panels.shape[1] <= 0:
                raise RuntimeError("each Ptera wing must have a nonempty panel grid")
            count += int(panels.shape[1])
    if count <= 0:
        raise RuntimeError("v5f requires at least one Ptera strip")
    return count


def _strip_map_to_panels(
    observer: PteraNativeLESPObserver,
    solver: UVPMHybridSolver,
) -> np.ndarray:
    lookup = {
        (int(ai), int(wi), int(si)): strip
        for strip, (ai, wi, si) in enumerate(
            zip(
                observer.strip_airplane_index,
                observer.strip_wing_index,
                observer.strip_spanwise_index,
                strict=True,
            )
        )
    }
    panel_strip: list[int] = []
    for airplane_index, airplane in enumerate(solver.current_airplanes):
        for wing_index, wing in enumerate(airplane.wings):
            for panel in np.ravel(wing.panels):
                spanwise = panel.local_spanwise_position
                if spanwise is None:
                    raise RuntimeError("Ptera panel has no spanwise topology")
                key = (airplane_index, wing_index, int(spanwise))
                if key not in lookup:
                    raise RuntimeError("observer and Ptera panel strip maps disagree")
                panel_strip.append(lookup[key])
    result = np.asarray(panel_strip, dtype=np.int64)
    if result.shape != (solver.num_panels,):
        raise RuntimeError("panel-to-strip map disagrees with Ptera panel count")
    return result


def _rear_bound_strengths(
    observer: PteraNativeLESPObserver,
    solver: UVPMHybridSolver,
) -> np.ndarray:
    values = np.empty(observer.operator.shape[0], dtype=float)
    for strip, (ai, wi, si) in enumerate(
        zip(
            observer.strip_airplane_index,
            observer.strip_wing_index,
            observer.strip_spanwise_index,
            strict=True,
        )
    ):
        wing = solver.current_airplanes[int(ai)].wings[int(wi)]
        panel = wing.panels[-1, int(si)]
        if panel.ring_vortex is None:
            raise RuntimeError("Ptera rear panel has no bound RingVortex")
        values[strip] = float(panel.ring_vortex.strength)
    return values


class NativeMaterialLEVTimeMarchSolver(UVPMHybridSolver):
    """Ptera-native v5f mechanical time marcher, never a scored paper model."""

    force_scoring_status: Final[str] = FORCE_SCORING_STATUS

    def __init__(
        self,
        unsteady_problem: Any,
        *,
        v5f_config: NativeMaterialLEVTimeMarchConfig,
        **uvpm_kwargs: Any,
    ) -> None:
        if not v5f_config.enabled:
            raise ValueError("disabled configurations must use the v5f factory")
        self.v5f_config = v5f_config
        self.v5f_step_reports: list[NativeMaterialLEVStepReport] = []
        self._v5f_pending: _PendingMaterialStep | None = None
        self._v5f_material_core_by_id: dict[int, float] = {}
        self._v5f_parent_wake_normal: np.ndarray | None = None
        self._v5f_parent_wake_front_strengths = np.empty(0, dtype=float)
        self._v5f_expected_wake_front_strengths: np.ndarray | None = None
        self._v5f_poisoned = False
        self._v5f_poison_reason: str | None = None
        super().__init__(unsteady_problem, **uvpm_kwargs)
        self.material_lev_state = PteraNativeMaterialLEVState(_count_strips(self))
        self._v5f_last_q = np.zeros(self.material_lev_state.ns, dtype=float)
        self._v5f_last_active = np.zeros(self.material_lev_state.ns, dtype=bool)

    def _finite_threshold_enabled(self) -> bool:
        return np.isfinite(float(self.v5f_config.lesp_critical))

    @property
    def v5f_poisoned(self) -> bool:
        """Whether an extension failure made this solver unsafe to resume."""

        return bool(self._v5f_poisoned)

    def _raise_if_poisoned(self) -> None:
        if self._v5f_poisoned:
            raise RuntimeError(
                "v5f solver is poisoned after a failed staged transaction: "
                f"{self._v5f_poison_reason}"
            )

    def _poison(self, error: BaseException) -> None:
        self._v5f_poisoned = True
        self._v5f_poison_reason = f"{type(error).__name__}: {error}"

    def _current_front_wake_strengths(self) -> np.ndarray:
        if self._current_step == 0:
            return np.empty(0, dtype=float)
        front: list[float] = []
        all_object_strengths: list[float] = []
        for airplane in self.current_airplanes:
            for wing in airplane.wings:
                wake = wing.wake_ring_vortices
                if wake is None or wake.shape[0] < 1:
                    raise RuntimeError("Ptera current step is missing its TE wake row")
                front.extend(float(vortex.strength) for vortex in wake[0])
                all_object_strengths.extend(
                    float(vortex.strength) for vortex in np.ravel(wake)
                )
        collapsed = np.asarray(self._current_wake_vortex_strengths, dtype=float)
        if not np.array_equal(collapsed, np.asarray(all_object_strengths)):
            raise RuntimeError("Ptera wake objects disagree with collapsed strengths")
        result = np.asarray(front, dtype=float)
        if result.shape != (self.material_lev_state.ns,):
            raise RuntimeError("Ptera TE front row disagrees with v5f strip count")
        return result

    def _core_radius_for_view(
        self,
        view: PteraMaterialLEVView,
        *,
        extra_core_by_id: dict[int, float] | None = None,
    ) -> np.ndarray:
        extra = {} if extra_core_by_id is None else extra_core_by_id
        core = np.empty(view.ring_id.size, dtype=float)
        for index, ring_id in enumerate(view.ring_id):
            key = int(ring_id)
            if key in extra:
                core[index] = float(extra[key])
            elif key in self._v5f_material_core_by_id:
                core[index] = self._v5f_material_core_by_id[key]
            else:
                raise RuntimeError("material ring has no frozen Eq. 25 core")
        return core

    def _material_view_velocity(
        self,
        points_gp1_m: np.ndarray,
        view: PteraMaterialLEVView,
        *,
        extra_core_by_id: dict[int, float] | None = None,
    ) -> np.ndarray:
        if view.ring_id.size == 0:
            return np.zeros_like(points_gp1_m)
        return eq25_velocity_from_ptera_material_rings(
            points_gp1_m,
            view.rings_gp1_m,
            view.gamma_m2_s,
            self._core_radius_for_view(view, extra_core_by_id=extra_core_by_id),
            surface_reflection_transform=(
                self.current_operating_point.surfaceReflect_T_act_GP1_CgP1
            ),
        )

    def _calculate_wake_wing_influences(self) -> None:
        self._raise_if_poisoned()
        super()._calculate_wake_wing_influences()
        if not self._finite_threshold_enabled():
            return
        if self._v5f_pending is not None:
            raise RuntimeError("previous v5f material proposal was not finalized")

        parent_wake = np.asarray(
            self._currentStackWakeWingInfluences__E,
            dtype=float,
        ).copy()
        self._v5f_parent_wake_normal = parent_wake
        current_front = self._current_front_wake_strengths()
        self._v5f_parent_wake_front_strengths = current_front.copy()
        if self._current_step > 0:
            expected = self._v5f_expected_wake_front_strengths
            if expected is None or not np.array_equal(current_front, expected):
                raise RuntimeError(
                    "current Ptera TE row violates the prior Eq. 9 ledger"
                )

        view = self.material_lev_state.material_view(
            reference_step=int(self._current_step)
        )
        if view.gamma_m2_s.size == 0:
            return
        history_velocity = self._material_view_velocity(
            np.asarray(self.stackCpp_GP1_CgP1, dtype=float),
            view,
        )
        history_normal = np.einsum(
            "ij,ij->i",
            history_velocity,
            self.stackUnitNormals_GP1,
        )
        combined = parent_wake + history_normal
        if not np.all(np.isfinite(combined)):
            raise FloatingPointError("historical material-LEV RHS is non-finite")
        self._currentStackWakeWingInfluences__E = combined

    def _registered_core_radii(
        self,
        observer: PteraNativeLESPObserver,
    ) -> np.ndarray:
        strip_width = np.linalg.norm(
            observer.leading_edges_gp1_m[:, 1] - observer.leading_edges_gp1_m[:, 0],
            axis=1,
        )
        birth_displacement = (
            observer.freestream_speed_m_s
            * float(self.v5f_config.lesp_critical)
            * float(self.delta_time)
            / np.sqrt(2.0)
        )
        configuration_scale = min(float(np.min(strip_width)), birth_displacement)
        core = np.full(
            strip_width.shape,
            float(self.v5f_config.core_radius_ratio) * configuration_scale,
            dtype=float,
        )
        if np.any(~np.isfinite(core)) or np.any(core <= 0.0):
            raise FloatingPointError("registered v5f Eq. 25 core is invalid")
        return core

    def _eq25_normal_columns(
        self,
        newborn_rings: np.ndarray,
        pseudovortex_rings: np.ndarray,
        core_radius: np.ndarray,
    ) -> np.ndarray:
        active_count = newborn_rings.shape[0]
        columns = np.empty((self.num_panels, active_count), dtype=float)
        reflection = self.current_operating_point.surfaceReflect_T_act_GP1_CgP1
        for column in range(active_count):
            points_velocity = eq25_velocity_from_ptera_material_rings(
                self.stackCpp_GP1_CgP1,
                np.stack((newborn_rings[column], pseudovortex_rings[column])),
                np.ones(2, dtype=float),
                np.full(2, core_radius[column], dtype=float),
                surface_reflection_transform=reflection,
            )
            columns[:, column] = np.einsum(
                "ij,ij->i",
                points_velocity,
                self.stackUnitNormals_GP1,
            )
        if not np.all(np.isfinite(columns)):
            raise FloatingPointError("v5f Eq. 25 birth influence is non-finite")
        return columns

    def _write_bound_strengths_transactionally(
        self,
        proposed: np.ndarray,
        parent: np.ndarray,
    ) -> None:
        panels = list(np.ravel(self.panels))
        old_ring = np.asarray(
            [float(panel.ring_vortex.strength) for panel in panels],
            dtype=float,
        )
        if not np.array_equal(old_ring, parent):
            raise RuntimeError("Ptera bound array and RingVortex objects disagree")
        try:
            self._current_bound_vortex_strengths = proposed.copy()
            for panel, strength in zip(panels, proposed, strict=True):
                panel.ring_vortex.strength = float(strength)
        except Exception:
            self._current_bound_vortex_strengths = parent.copy()
            for panel, strength in zip(panels, old_ring, strict=True):
                panel.ring_vortex.strength = float(strength)
            raise

    def _first_or_restart_aft_edges(
        self,
        observer: PteraNativeLESPObserver,
        a0_pre: np.ndarray,
        new_sheet: np.ndarray,
    ) -> np.ndarray:
        """Return the unchanged v5f 2012 first/restart birth geometry.

        ``new_sheet`` is accepted by the hook contract but deliberately does
        not alter the legacy arithmetic.  The material state consumes these
        aft edges only where ``new_sheet`` is true; continuous shedding stays
        on Hirato Eq. 7's one-third split.
        """

        del new_sheet
        local_alpha = np.arctan2(
            -observer.chord_tangent_gp1[:, 2],
            observer.chord_tangent_gp1[:, 0],
        )
        displacement_2d = first_lev_displacement_ramesh_2d(
            observer.freestream_speed_m_s,
            a0_pre,
            local_alpha,
            float(self.delta_time),
        )
        displacement = embed_chord_normal_displacement(
            displacement_2d,
            observer.chord_tangent_gp1,
            observer.suction_normal_gp1,
        )
        return observer.leading_edges_gp1_m + displacement[:, None, :]

    def _calculate_vortex_strengths(self) -> None:
        self._raise_if_poisoned()
        super()._calculate_vortex_strengths()
        if not self._finite_threshold_enabled():
            return
        if self._v5f_parent_wake_normal is None:
            raise RuntimeError("v5f parent wake ledger was not staged")

        observer = build_ptera_native_lesp_observer(self)
        if observer.operator.shape[0] != self.material_lev_state.ns:
            raise RuntimeError("material-state and Ptera strip counts disagree")
        parent_gamma = np.asarray(
            self._current_bound_vortex_strengths,
            dtype=float,
        ).copy()
        a0_pre = observer.observe(parent_gamma)
        threshold = float(self.v5f_config.lesp_critical)
        active = np.abs(a0_pre) > threshold
        new_sheet = active & ~self._v5f_last_active
        material_snapshot_before = self.material_lev_state.snapshot()

        first_aft_edges = self._first_or_restart_aft_edges(
            observer,
            a0_pre,
            new_sheet,
        )
        proposal = self.material_lev_state.propose_shed(
            step=int(self._current_step),
            leading_edges_gp1_m=observer.leading_edges_gp1_m,
            active=active,
            new_sheet=new_sheet,
            first_aft_edges_gp1_m=first_aft_edges,
        )

        try:
            active_indices = proposal.active_strip_indices
            core_by_strip = self._registered_core_radii(observer)
            created_core_by_id = {
                int(ring_id): float(core_by_strip[int(strip)])
                for ring_id, strip in zip(
                    proposal.created_ids,
                    active_indices,
                    strict=True,
                )
            }
            leading = observer.leading_edges_gp1_m[active_indices]
            rear = observer.rear_bound_aft_edges_gp1_m[active_indices]
            pseudovortex_rings = np.stack(
                (leading[:, 0], leading[:, 1], rear[:, 0], rear[:, 1]),
                axis=1,
            )
            if active_indices.size:
                fixed_view = proposal.candidate_material_view()
                fixed_material_velocity = self._material_view_velocity(
                    np.asarray(self.stackCpp_GP1_CgP1, dtype=float),
                    fixed_view,
                    extra_core_by_id=created_core_by_id,
                )
                fixed_material_normal = np.einsum(
                    "ij,ij->i",
                    fixed_material_velocity,
                    self.stackUnitNormals_GP1,
                )
                influence = self._eq25_normal_columns(
                    proposal.newborn_ptera_rings_gp1_m,
                    pseudovortex_rings,
                    core_by_strip[active_indices],
                )
                target = threshold * np.sign(a0_pre[active])
                rhs = (
                    -self._v5f_parent_wake_normal
                    - fixed_material_normal
                    - self._currentStackFreestreamWingInfluences__E
                )
                augmented = solve_ptera_native_augmented_system(
                    self._currentGridWingWingInfluences__E,
                    rhs,
                    lev_unit_normal_influence=influence,
                    lesp_operator=observer.operator[active],
                    lesp_target=target,
                    active=True,
                )
            else:
                pseudovortex_rings = np.empty((0, 4, 3), dtype=float)
                augmented = solve_ptera_native_augmented_system(
                    self._currentGridWingWingInfluences__E,
                    -self._currentStackWakeWingInfluences__E
                    - self._currentStackFreestreamWingInfluences__E,
                    active=False,
                )
            q = np.zeros(observer.operator.shape[0], dtype=float)
            q[active] = augmented.gamma_lev
            staged_view = proposal.candidate_material_view(
                created_strengths_m2_s=q[active],
            )
            proposed_gamma = augmented.gamma_bound.copy()
            a0_post = observer.observe(proposed_gamma)
            if np.any(active):
                target_residual = a0_post[active] - (
                    threshold * np.sign(a0_pre[active])
                )
                if (
                    not np.all(np.isfinite(target_residual))
                    or float(np.max(np.abs(target_residual), initial=0.0))
                    > AUGMENTED_RESIDUAL_ATOL
                ):
                    raise FloatingPointError(
                        "v5f material-step LESP target did not close"
                    )
                self._write_bound_strengths_transactionally(
                    proposed_gamma,
                    parent_gamma,
                )
            elif not np.array_equal(proposed_gamma, parent_gamma):
                raise FloatingPointError("inactive material step changed Ptera")
        except Exception as error:
            if not proposal.is_finalized:
                proposal.abort()
            self._poison(error)
            raise

        self._v5f_pending = _PendingMaterialStep(
            observer=observer,
            proposal=proposal,
            active=active.copy(),
            q_by_strip_m2_s=q,
            a0_pre=a0_pre,
            a0_post=a0_post,
            core_radius_by_strip_m=core_by_strip,
            augmented_solution=augmented,
            staged_view=staged_view,
            pseudovortex_rings_gp1_m=pseudovortex_rings,
            parent_gamma_bound=parent_gamma,
            material_snapshot_before=material_snapshot_before,
            created_ring_ids=proposal.created_ids,
            new_sheet=new_sheet.copy(),
            parent_wake_front_strengths_m2_s=(
                self._v5f_parent_wake_front_strengths.copy()
            ),
        )

    def _pending_material_velocity(
        self,
        points_gp1_m: np.ndarray,
    ) -> np.ndarray:
        pending = self._v5f_pending
        if pending is None:
            return np.zeros_like(points_gp1_m)
        view = pending.staged_view
        active_core = {
            int(ring_id): float(pending.core_radius_by_strip_m[int(strip)])
            for ring_id, strip in zip(
                pending.proposal.created_ids,
                pending.proposal.active_strip_indices,
                strict=True,
            )
        }
        reflection = self.current_operating_point.surfaceReflect_T_act_GP1_CgP1
        material = self._material_view_velocity(
            points_gp1_m,
            view,
            extra_core_by_id=active_core,
        )
        active_indices = pending.proposal.active_strip_indices
        if active_indices.size:
            pseudo = eq25_velocity_from_ptera_material_rings(
                points_gp1_m,
                pending.pseudovortex_rings_gp1_m,
                pending.q_by_strip_m2_s[active_indices],
                pending.core_radius_by_strip_m[active_indices],
                surface_reflection_transform=reflection,
            )
        else:
            pseudo = np.zeros_like(points_gp1_m)
        return material + pseudo

    def calculate_solution_velocity(
        self,
        stackP_GP1_CgP1: np.ndarray | Sequence[Sequence[float | int]],
        bound_singularity_counts: np.ndarray | None = None,
        wake_singularity_counts: np.ndarray | None = None,
    ) -> np.ndarray:
        parent = super().calculate_solution_velocity(
            stackP_GP1_CgP1,
            bound_singularity_counts=bound_singularity_counts,
            wake_singularity_counts=wake_singularity_counts,
        )
        pending = self._v5f_pending
        if pending is None:
            return parent
        if (
            pending.staged_view.gamma_m2_s.size == 0
            and pending.pseudovortex_rings_gp1_m.shape[0] == 0
        ):
            return parent
        points = _finite_array(
            stackP_GP1_CgP1,
            name="stackP_GP1_CgP1",
            ndim=2,
        )
        extension = self._pending_material_velocity(points)
        result = np.asarray(parent, dtype=float) + extension
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("v5f solution velocity is non-finite")
        return result

    def _calculate_loads(self) -> None:
        self._raise_if_poisoned()
        super()._calculate_loads()
        if not self._finite_threshold_enabled():
            return
        pending = self._v5f_pending
        if pending is None:
            raise RuntimeError("v5f material solve was not staged before loads")

        panel_strip = _strip_map_to_panels(pending.observer, self)
        q_panel = pending.q_by_strip_m2_s[panel_strip]
        q_previous_panel = self._v5f_last_q[panel_strip]
        active_panel = pending.active[panel_strip]
        try:
            load_delta = calculate_live_native_surface_release_load_delta(
                self,
                gamma_lev_current_m2_s=q_panel,
                gamma_lev_previous_m2_s=(
                    None if self._current_step == 0 else q_previous_panel
                ),
                active_current=active_panel,
            )
            panels = list(np.ravel(self.panels))
            parent_force = np.asarray(
                [np.asarray(panel.forces_GP1, dtype=float) for panel in panels]
            )
            parent_moment = np.asarray(
                [np.asarray(panel.moments_GP1_CgP1, dtype=float) for panel in panels]
            )
            correction = load_delta.correction
            delta_force = np.asarray(
                correction["delta_total_force_gp1_n"],
                dtype=float,
            )
            delta_moment = np.asarray(
                correction["delta_total_moment_gp1_cgp1_nm"],
                dtype=float,
            )
            if np.any(delta_force != 0.0) or np.any(delta_moment != 0.0):
                _functions.process_solver_loads(
                    self,
                    parent_force + delta_force,
                    parent_moment + delta_moment,
                )
            pending.load_delta = load_delta
        except Exception as error:
            if not pending.proposal.is_finalized:
                pending.proposal.abort()
            self._poison(error)
            raise

    def _patch_next_wake_front(
        self,
        pending: _PendingMaterialStep,
    ) -> tuple[np.ndarray, np.ndarray, list[tuple[Any, float]]]:
        rear_bound = _rear_bound_strengths(pending.observer, self)
        expected = rear_bound + pending.q_by_strip_m2_s
        actual = np.empty_like(expected)
        restore: list[tuple[Any, float]] = []
        if self._current_step >= self.num_steps - 1:
            return rear_bound, np.empty(0, dtype=float), restore
        next_problem = self.steady_problems[self._current_step + 1]
        for strip, (ai, wi, si) in enumerate(
            zip(
                pending.observer.strip_airplane_index,
                pending.observer.strip_wing_index,
                pending.observer.strip_spanwise_index,
                strict=True,
            )
        ):
            next_wing = next_problem.airplanes[int(ai)].wings[int(wi)]
            wake = next_wing.wake_ring_vortices
            if wake is None or wake.shape[0] < 1:
                raise RuntimeError("Ptera did not create the next TE wake row")
            vortex = wake[0, int(si)]
            old_strength = float(vortex.strength)
            if old_strength != rear_bound[strip]:
                raise RuntimeError("Ptera parent TE row is not current rear bound")
            restore.append((vortex, old_strength))
            vortex.strength = float(expected[strip])
            actual[strip] = float(vortex.strength)
        return rear_bound, actual, restore

    def _populate_next_airplanes_wake(self) -> None:
        self._raise_if_poisoned()
        if not self._finite_threshold_enabled():
            return super()._populate_next_airplanes_wake()
        pending = self._v5f_pending
        if pending is None:
            raise RuntimeError("v5f material state is missing before wake commit")

        staged = pending.staged_view
        if staged.rings_gp1_m.shape[0]:
            flat_vertices = staged.rings_gp1_m.reshape(-1, 3)
            vertex_velocity = self.calculate_solution_velocity(flat_vertices).reshape(
                staged.rings_gp1_m.shape
            )
        else:
            vertex_velocity = np.empty((0, 4, 3), dtype=float)
        if not np.all(np.isfinite(vertex_velocity)):
            raise FloatingPointError("material-vertex velocity is non-finite")

        super()._populate_next_airplanes_wake()
        restore: list[tuple[Any, float]] = []
        try:
            rear_bound, next_front, restore = self._patch_next_wake_front(pending)
            if next_front.size:
                eq9_residual = next_front - (rear_bound + pending.q_by_strip_m2_s)
                eq9_max = float(np.max(np.abs(eq9_residual), initial=0.0))
                if eq9_max > 1.0e-12:
                    raise FloatingPointError("v5f Eq. 9 TE ledger did not close")
            else:
                eq9_max = 0.0
            snapshot, _ = pending.proposal.commit_and_convect(
                pending.q_by_strip_m2_s[pending.active],
                vertex_velocity,
                dt_s=float(self.delta_time),
            )
        except Exception as error:
            for vortex, old_strength in restore:
                vortex.strength = old_strength
            if not pending.proposal.is_finalized:
                pending.proposal.abort()
            self._poison(error)
            raise

        for ring_id, strip in zip(
            pending.proposal.created_ids,
            pending.proposal.active_strip_indices,
            strict=True,
        ):
            self._v5f_material_core_by_id[int(ring_id)] = float(
                pending.core_radius_by_strip_m[int(strip)]
            )
        self._v5f_last_q = pending.q_by_strip_m2_s.copy()
        self._v5f_last_active = pending.active.copy()
        self._v5f_expected_wake_front_strengths = (
            next_front.copy() if next_front.size else None
        )
        self.v5f_step_reports.append(
            NativeMaterialLEVStepReport(
                step=int(self._current_step),
                active=pending.active.copy(),
                q_by_strip_m2_s=pending.q_by_strip_m2_s.copy(),
                a0_pre=pending.a0_pre.copy(),
                a0_post=pending.a0_post.copy(),
                core_radius_by_strip_m=pending.core_radius_by_strip_m.copy(),
                augmented_solution=pending.augmented_solution,
                load_delta=pending.load_delta,
                material_snapshot_before=pending.material_snapshot_before,
                created_ring_ids=pending.created_ring_ids.copy(),
                new_sheet=pending.new_sheet.copy(),
                parent_wake_front_strengths_m2_s=(
                    pending.parent_wake_front_strengths_m2_s.copy()
                ),
                rear_bound_strengths_m2_s=rear_bound.copy(),
                next_wake_front_strengths_m2_s=next_front.copy(),
                material_snapshot=snapshot,
                eq9_max_abs_m2_s=eq9_max,
                force_scoring_status=FORCE_SCORING_STATUS,
                continuation_status=CONTINUATION_STATUS,
            )
        )
        self._v5f_pending = None


class NativeMaterialLEVLDVMV25BirthTimeMarchSolver(NativeMaterialLEVTimeMarchSolver):
    """Isolated v5f2 first/restart birth-law ablation.

    The augmented solve, Hirato Eq. 7 continuation, loads, TE ledger,
    convection and registered v5f core rule are inherited without change.
    Only :meth:`_first_or_restart_aft_edges` is replaced.
    """

    def __init__(
        self,
        unsteady_problem: Any,
        *,
        v5f_config: NativeMaterialLEVTimeMarchConfig,
        **uvpm_kwargs: Any,
    ) -> None:
        self.v5f2_birth_velocity_ledgers: list[
            PteraMappedLDVMV25BirthVelocityLedger
        ] = []
        super().__init__(
            unsteady_problem,
            v5f_config=v5f_config,
            **uvpm_kwargs,
        )

    def _current_parent_te_wake_velocity(
        self,
        points_gp1_m: np.ndarray,
    ) -> np.ndarray:
        """Return only the TE-origin wake present in Ptera at ``t_n``.

        This excludes bound and material-LEV rings.  It also cannot include
        the provisional current-step TEV used before the LE placement in the
        LDVM-v2.5 Fortran time layer because Ptera has no such ring at this
        hook.  That transfer limitation is frozen in every birth ledger.
        """

        points = _finite_array(points_gp1_m, name="points_gp1_m", ndim=2)
        if points.shape[1:] != (3,):
            raise ValueError("points_gp1_m must have shape (N,3)")
        strengths = np.asarray(self._current_wake_vortex_strengths, dtype=float)
        if int(self._current_step) == 0 or strengths.size == 0:
            return np.zeros_like(points)
        singularity_counts = np.zeros(4, dtype=np.int64)
        velocity = _aerodynamics_functions.collapsed_velocities_from_ring_vortices(
            stackP_GP1_CgP1=points,
            stackBrrvp_GP1_CgP1=self._currentStackBrwrvp_GP1_CgP1,
            stackFrrvp_GP1_CgP1=self._currentStackFrwrvp_GP1_CgP1,
            stackFlrvp_GP1_CgP1=self._currentStackFlwrvp_GP1_CgP1,
            stackBlrvp_GP1_CgP1=self._currentStackBlwrvp_GP1_CgP1,
            strengths=strengths,
            r_c0s=self._currentStackWakeRc0s,
            singularity_counts=singularity_counts,
            ages=self._current_wake_vortex_ages,
            nu=self.current_operating_point.nu,
        )
        reflection = self.current_operating_point.surfaceReflect_T_act_GP1_CgP1
        if reflection is not None:
            reflected_points = _transformations.apply_T_to_vectors(
                reflection,
                points,
                has_point=True,
            )
            image_velocity = (
                _aerodynamics_functions.collapsed_velocities_from_ring_vortices(
                    stackP_GP1_CgP1=reflected_points,
                    stackBrrvp_GP1_CgP1=self._currentStackBrwrvp_GP1_CgP1,
                    stackFrrvp_GP1_CgP1=self._currentStackFrwrvp_GP1_CgP1,
                    stackFlrvp_GP1_CgP1=self._currentStackFlwrvp_GP1_CgP1,
                    stackBlrvp_GP1_CgP1=self._currentStackBlwrvp_GP1_CgP1,
                    strengths=strengths,
                    r_c0s=self._currentStackWakeRc0s,
                    singularity_counts=singularity_counts,
                    ages=self._current_wake_vortex_ages,
                    nu=self.current_operating_point.nu,
                )
            )
            velocity = velocity + _transformations.apply_T_to_vectors(
                reflection,
                image_velocity,
                has_point=False,
            )
        if not np.all(np.isfinite(velocity)):
            raise FloatingPointError("Ptera parent TE-wake velocity is non-finite")
        return np.asarray(velocity, dtype=float)

    def _current_leading_edge_apparent_movement(
        self,
        observer: PteraNativeLESPObserver,
    ) -> np.ndarray:
        """Return Ptera-sign apparent LE motion at the current time layer."""

        current = np.asarray(observer.leading_edges_gp1_m, dtype=float)
        if int(self._current_step) == 0:
            return np.zeros_like(current)
        previous_problem = self.steady_problems[int(self._current_step) - 1]
        previous = np.empty_like(current)
        for strip, (ai, wi, si) in enumerate(
            zip(
                observer.strip_airplane_index,
                observer.strip_wing_index,
                observer.strip_spanwise_index,
                strict=True,
            )
        ):
            previous_panel = (
                previous_problem.airplanes[int(ai)].wings[int(wi)].panels[0, int(si)]
            )
            previous[strip, 0] = np.asarray(
                previous_panel.Flpp_GP1_CgP1,
                dtype=float,
            )
            previous[strip, 1] = np.asarray(
                previous_panel.Frpp_GP1_CgP1,
                dtype=float,
            )
        movement = -(current - previous) / float(self.delta_time)
        if not np.all(np.isfinite(movement)):
            raise FloatingPointError("leading-edge apparent movement is non-finite")
        return movement

    def _first_or_restart_aft_edges(
        self,
        observer: PteraNativeLESPObserver,
        a0_pre: np.ndarray,
        new_sheet: np.ndarray,
    ) -> np.ndarray:
        """Map the mature LDVM half-step law to the staged Ptera edge field."""

        del a0_pre
        leading = np.asarray(observer.leading_edges_gp1_m, dtype=float)
        flat_points = leading.reshape(-1, 3)
        freestream = np.broadcast_to(
            np.asarray(self.current_operating_point.vInf_GP1__E, dtype=float),
            leading.shape,
        ).copy()
        parent_te_wake = self._current_parent_te_wake_velocity(flat_points).reshape(
            leading.shape
        )
        committed = self.material_lev_state.material_view(
            reference_step=int(self._current_step)
        )
        committed_material = self._material_view_velocity(
            flat_points,
            committed,
        ).reshape(leading.shape)
        apparent_movement = self._current_leading_edge_apparent_movement(observer)
        span_vector = leading[:, 1] - leading[:, 0]
        span_tangent = (
            span_vector
            / np.linalg.norm(
                span_vector,
                axis=1,
            )[:, None]
        )
        ledger = assemble_ptera_mapped_ldvm_v25_local_edge_velocity(
            step=int(self._current_step),
            new_sheet=new_sheet,
            freestream_m_s=freestream,
            current_parent_te_wake_induced_m_s=parent_te_wake,
            committed_old_material_induced_m_s=committed_material,
            apparent_movement_m_s=apparent_movement,
            span_tangent_gp1=span_tangent,
        )
        displacement = first_lev_displacement_ramesh_2014_ldvm_v25(
            ledger.total_local_edge_velocity_m_s,
            float(self.delta_time),
        )
        self.v5f2_birth_velocity_ledgers.append(ledger)
        return leading + displacement


def make_fluxv_v5f_native_time_march_solver(
    unsteady_problem: Any,
    *,
    config: NativeMaterialLEVTimeMarchConfig = DEFAULT_NATIVE_TIME_MARCH_CONFIG,
    **uvpm_kwargs: Any,
) -> UVPMHybridSolver:
    """Return the exact parent or the finite-threshold v5f mechanical subclass."""

    if not isinstance(config, NativeMaterialLEVTimeMarchConfig):
        raise TypeError("config must be NativeMaterialLEVTimeMarchConfig")
    if not bool(config.enabled):
        return UVPMHybridSolver(unsteady_problem, **uvpm_kwargs)
    return NativeMaterialLEVTimeMarchSolver(
        unsteady_problem,
        v5f_config=config,
        **uvpm_kwargs,
    )


def make_fluxv_v5f2_ldvm_v25_birth_time_march_solver(
    unsteady_problem: Any,
    *,
    config: NativeMaterialLEVTimeMarchConfig = DEFAULT_NATIVE_TIME_MARCH_CONFIG,
    **uvpm_kwargs: Any,
) -> UVPMHybridSolver:
    """Select the isolated 2014/LDVM-v2.5 Ptera-mapped birth ablation.

    Disabled mode remains the exact UVPM parent.  Enabling this explicit
    factory changes only first/restart aft-edge placement; the legacy v5f
    factory and default 2012 path are untouched.
    """

    if not isinstance(config, NativeMaterialLEVTimeMarchConfig):
        raise TypeError("config must be NativeMaterialLEVTimeMarchConfig")
    if not bool(config.enabled):
        return UVPMHybridSolver(unsteady_problem, **uvpm_kwargs)
    return NativeMaterialLEVLDVMV25BirthTimeMarchSolver(
        unsteady_problem,
        v5f_config=config,
        **uvpm_kwargs,
    )


__all__ = [
    "DEFAULT_NATIVE_TIME_MARCH_CONFIG",
    "FORCE_SCORING_STATUS",
    "PTERA_LDVM_V25_BIRTH_MAPPING_STATUS",
    "PteraMappedLDVMV25BirthVelocityLedger",
    "NativeMaterialLEVLDVMV25BirthTimeMarchSolver",
    "NativeMaterialLEVStepReport",
    "NativeMaterialLEVTimeMarchConfig",
    "NativeMaterialLEVTimeMarchSolver",
    "REGISTERED_CORE_RADIUS_RATIOS",
    "eq25_velocity_from_ptera_material_rings",
    "assemble_ptera_mapped_ldvm_v25_local_edge_velocity",
    "make_fluxv_v5f2_ldvm_v25_birth_time_march_solver",
    "make_fluxv_v5f_native_time_march_solver",
]
