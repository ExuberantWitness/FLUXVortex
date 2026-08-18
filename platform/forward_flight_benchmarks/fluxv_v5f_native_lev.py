"""Fail-closed first stage of a Ptera-native material-LEV extension.

This module deliberately keeps PteraSoftware as the sole owner of the bound
ring-vortex AIC, right-hand side, wake, and Kutta--Joukowski plus
``dGamma/dt`` load calculation.  Stage 1 provides:

* an exactly passive solver path for proving reduction to
  :class:`fluxvortex.solver.UVPMHybridSolver`; and
* a small, independently testable augmented-system helper that consumes a
  *Ptera-produced* AIC and a unit-LEV influence column in the same convention.

Stage 2A adds a transactional single-step probe.  It observes LESP from the
actual Ptera lattice, builds newborn-LEV and Hirato pseudovortex influence
columns with Ptera's own ring kernel, solves the literal augmented system, and
can update only the current bound strengths.  It does not time-march or commit
the proposed free vortices.

The material wake and the corresponding Ptera load-ledger corrections are not
implemented here.  A finite LESP activation threshold therefore fails closed
instead of silently switching to a different vortex kernel or force model.
This is intentional: results from this stage must not be scored against force
measurements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Sequence

import numpy as np
from pterasoftware import _aerodynamics_functions, _transformations

from fluxvortex.solver import UVPMHybridSolver

from claim_runtime.hirato_equations import (
    embed_chord_normal_displacement,
    first_lev_displacement_ramesh_2d,
    lesp_eq6,
    preconstraint_shed_mask,
)


FORCE_SCORING_STATUS: Final = "blocked_until_material_wake_and_ptera_load_corrections"
AUGMENTED_RESIDUAL_ATOL: Final = 1.0e-10


@dataclass(frozen=True)
class NativeMaterialLEVConfig:
    """Configuration for the v5f Stage-1 solver factory.

    ``lesp_critical=np.inf`` is the only enabled configuration accepted in
    Stage 1.  It makes activation unreachable and leaves the material history
    pristine.  Disabled configurations are ignored by the factory and produce
    a direct :class:`UVPMHybridSolver`, even if the threshold is finite.
    """

    enabled: bool = False
    lesp_critical: float = np.inf

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, (bool, np.bool_)):
            raise TypeError("enabled must be boolean")
        threshold = float(self.lesp_critical)
        if np.isnan(threshold) or threshold <= 0.0:
            raise ValueError("lesp_critical must be positive and not NaN")


DEFAULT_NATIVE_MATERIAL_LEV_CONFIG: Final = NativeMaterialLEVConfig()


@dataclass(frozen=True)
class PteraNativeAugmentedSolution:
    """Result and residual ledger for the exact block system used by v5f.

    ``augmented_matrix`` is either the supplied Ptera bound AIC (inactive
    path) or the literal ``[A B; H 0]`` matrix (active path).  No ridge,
    circulation cap, least-squares fallback, or sign conversion is applied.
    """

    gamma_bound: np.ndarray
    gamma_lev: np.ndarray
    no_penetration_residual: np.ndarray
    lesp_constraint_residual: np.ndarray
    augmented_matrix: np.ndarray
    augmented_rhs: np.ndarray
    used_augmented_system: bool
    reduction_reason: str | None

    @property
    def no_penetration_max_abs(self) -> float:
        """Maximum absolute no-penetration residual."""

        return float(np.max(np.abs(self.no_penetration_residual), initial=0.0))

    @property
    def lesp_constraint_max_abs(self) -> float:
        """Maximum absolute LESP-constraint residual."""

        return float(np.max(np.abs(self.lesp_constraint_residual), initial=0.0))


@dataclass(frozen=True)
class PteraNativeLESPObserver:
    """Eq. 6 observation operator assembled in Ptera's panel order."""

    operator: np.ndarray
    leading_panel_indices: np.ndarray
    chord_m: np.ndarray
    delta_x_front_m: np.ndarray
    freestream_speed_m_s: float
    leading_edges_gp1_m: np.ndarray
    rear_bound_aft_edges_gp1_m: np.ndarray
    chord_tangent_gp1: np.ndarray
    suction_normal_gp1: np.ndarray
    ptera_core_radius_m: np.ndarray
    strip_airplane_index: np.ndarray
    strip_wing_index: np.ndarray
    strip_spanwise_index: np.ndarray

    def observe(self, gamma_bound: np.ndarray | Sequence[float]) -> np.ndarray:
        """Apply the sparse linear Eq. 6 operator to bound circulation."""

        gamma = _finite_float_array(gamma_bound, name="gamma_bound", ndim=1)
        if gamma.shape != (self.operator.shape[1],):
            raise ValueError("gamma_bound length must match the Ptera panel count")
        return self.operator @ gamma

    def direct_eq6(self, gamma_bound: np.ndarray | Sequence[float]) -> np.ndarray:
        """Evaluate Eq. 6 directly from the same Ptera leading-panel state."""

        gamma = _finite_float_array(gamma_bound, name="gamma_bound", ndim=1)
        if gamma.shape != (self.operator.shape[1],):
            raise ValueError("gamma_bound length must match the Ptera panel count")
        return lesp_eq6(
            gamma[self.leading_panel_indices],
            self.freestream_speed_m_s,
            self.chord_m,
            self.delta_x_front_m,
        )


@dataclass(frozen=True)
class PteraNativeBirthInfluence:
    """Ptera-kernel unit influence ledger for nascent LEV rings."""

    active_strip_indices: np.ndarray
    newborn_rings_gp1_m: np.ndarray
    pseudovortex_rings_gp1_m: np.ndarray
    newborn_unit_normal_influence: np.ndarray
    pseudovortex_unit_normal_influence: np.ndarray
    combined_unit_normal_influence: np.ndarray
    singularity_counts: np.ndarray


@dataclass(frozen=True)
class NativeActiveProbeReport:
    """Transactional, bound-only result of one native active probe."""

    lesp_critical: float
    parent_gamma_bound: np.ndarray
    proposed_gamma_bound: np.ndarray
    gamma_lev_by_strip: np.ndarray
    a0_parent: np.ndarray
    a0_post: np.ndarray
    active: np.ndarray
    observer: PteraNativeLESPObserver
    birth_influence: PteraNativeBirthInfluence
    augmented_solution: PteraNativeAugmentedSolution
    committed_bound_strengths: bool
    force_scoring_status: str
    state_scope: str


def _finite_float_array(value: Any, *, name: str, ndim: int) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _unit_rows(vectors: np.ndarray, *, name: str) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
        raise ValueError(f"{name} contains zero or non-finite vectors")
    return vectors / norms[:, None]


def build_ptera_native_lesp_observer(
    solver: UVPMHybridSolver,
) -> PteraNativeLESPObserver:
    """Build Hirato Eq. 6's ``H`` from the current Ptera lattice.

    Panel indices are obtained by traversing ``current_airplanes`` exactly as
    Ptera's ``_collapse_geometry`` does: airplane, wing, then C-order ravel of
    each ``(chord, span)`` panel grid.  Hirato's section chord and first-cell
    extent come from the physical panel corner geometry; Ptera's advected aft
    bound-ring extension is deliberately excluded.  No alternate bound AIC
    lattice is constructed.
    """

    required = (
        "current_airplanes",
        "current_operating_point",
        "panels",
        "stackBrbrvp_GP1_CgP1",
        "stackFrbrvp_GP1_CgP1",
        "stackFlbrvp_GP1_CgP1",
        "stackBlbrvp_GP1_CgP1",
        "_currentStackBoundRc0s",
    )
    if any(getattr(solver, name, None) is None for name in required):
        raise RuntimeError("Ptera current-step geometry has not been collapsed")

    ptera_panels = list(np.ravel(solver.panels))
    panel_count = len(ptera_panels)
    if panel_count == 0:
        raise RuntimeError("Ptera current step contains no bound panels")
    br = _finite_float_array(
        solver.stackBrbrvp_GP1_CgP1,
        name="stackBrbrvp_GP1_CgP1",
        ndim=2,
    )
    fr = _finite_float_array(
        solver.stackFrbrvp_GP1_CgP1,
        name="stackFrbrvp_GP1_CgP1",
        ndim=2,
    )
    fl = _finite_float_array(
        solver.stackFlbrvp_GP1_CgP1,
        name="stackFlbrvp_GP1_CgP1",
        ndim=2,
    )
    bl = _finite_float_array(
        solver.stackBlbrvp_GP1_CgP1,
        name="stackBlbrvp_GP1_CgP1",
        ndim=2,
    )
    if any(array.shape != (panel_count, 3) for array in (br, fr, fl, bl)):
        raise RuntimeError("Ptera bound-ring arrays do not match its panel order")
    core_radius = _finite_float_array(
        solver._currentStackBoundRc0s,
        name="_currentStackBoundRc0s",
        ndim=1,
    )
    if core_radius.shape != (panel_count,) or np.any(core_radius < 0.0):
        raise RuntimeError("Ptera bound core radii do not match its panel order")

    leading_indices: list[int] = []
    chord_values: list[float] = []
    front_values: list[float] = []
    leading_edges: list[np.ndarray] = []
    rear_aft_edges: list[np.ndarray] = []
    chord_tangents: list[np.ndarray] = []
    suction_normals: list[np.ndarray] = []
    source_core_radii: list[float] = []
    airplane_indices: list[int] = []
    wing_indices: list[int] = []
    spanwise_indices: list[int] = []

    global_offset = 0
    traversed_panels: list[Any] = []
    for airplane_index, airplane in enumerate(solver.current_airplanes):
        for wing_index, wing in enumerate(airplane.wings):
            panel_grid = np.asarray(wing.panels, dtype=object)
            if panel_grid.ndim != 2 or panel_grid.size == 0:
                raise RuntimeError("each Ptera wing must have a 2D panel grid")
            chord_count, span_count = panel_grid.shape
            traversed_panels.extend(np.ravel(panel_grid))
            for spanwise_index in range(span_count):
                indices = global_offset + np.arange(chord_count) * span_count
                indices = indices + spanwise_index
                leading_index = int(indices[0])
                rear_index = int(indices[-1])
                leading_panel = panel_grid[0, spanwise_index]
                rear_panel = panel_grid[-1, spanwise_index]
                if not bool(leading_panel.is_leading_edge):
                    raise RuntimeError("Ptera leading-panel topology is inconsistent")
                if not bool(rear_panel.is_trailing_edge):
                    raise RuntimeError("Ptera trailing-panel topology is inconsistent")

                le_left = np.asarray(leading_panel.Flpp_GP1_CgP1, dtype=float).copy()
                le_right = np.asarray(leading_panel.Frpp_GP1_CgP1, dtype=float).copy()
                te_left = np.asarray(rear_panel.Blpp_GP1_CgP1, dtype=float)
                te_right = np.asarray(rear_panel.Brpp_GP1_CgP1, dtype=float)
                chord_vector = 0.5 * ((te_left - le_left) + (te_right - le_right))
                front_chord_vector = 0.5 * (
                    (np.asarray(leading_panel.Blpp_GP1_CgP1, dtype=float) - le_left)
                    + (np.asarray(leading_panel.Brpp_GP1_CgP1, dtype=float) - le_right)
                )
                physical_chord = float(np.linalg.norm(chord_vector))
                physical_front_extent = float(np.linalg.norm(front_chord_vector))
                if (
                    not np.isfinite(physical_chord)
                    or not np.isfinite(physical_front_extent)
                    or physical_chord <= 0.0
                    or physical_front_extent <= 0.0
                    or physical_front_extent > physical_chord
                ):
                    raise RuntimeError(
                        "Ptera panel geometry has invalid physical chord"
                    )
                span_vector = le_right - le_left
                chord_tangent = _unit_rows(
                    chord_vector[None, :], name="strip chord tangent"
                )[0]
                span_tangent = _unit_rows(
                    span_vector[None, :], name="leading-edge span tangent"
                )[0]
                suction_normal = _unit_rows(
                    np.cross(chord_tangent, span_tangent)[None, :],
                    name="strip suction normal",
                )[0]
                panel_normal = np.asarray(leading_panel.unitNormal_GP1, dtype=float)
                if float(np.dot(suction_normal, panel_normal)) < 0.0:
                    suction_normal = -suction_normal
                chord_tangent = _unit_rows(
                    np.cross(span_tangent, suction_normal)[None, :],
                    name="orthogonal strip chord tangent",
                )[0]

                leading_indices.append(leading_index)
                # Hirato Eq. 6 uses the physical section chord.  Ptera's
                # rearmost bound ring contains an advective aft extension, so
                # summing bound-ring extents would spuriously couple LESP to
                # the time step and wing motion.
                chord_values.append(physical_chord)
                front_values.append(physical_front_extent)
                leading_edges.append(np.stack((le_left, le_right)))
                rear_aft_edges.append(np.stack((br[rear_index], bl[rear_index])))
                chord_tangents.append(chord_tangent)
                suction_normals.append(suction_normal)
                source_core_radii.append(float(core_radius[leading_index]))
                airplane_indices.append(airplane_index)
                wing_indices.append(wing_index)
                spanwise_indices.append(spanwise_index)
            global_offset += panel_grid.size

    if global_offset != panel_count or any(
        actual is not expected
        for actual, expected in zip(traversed_panels, ptera_panels, strict=True)
    ):
        raise RuntimeError("local panel traversal differs from Ptera flatten order")

    leading_panel_indices = np.asarray(leading_indices, dtype=int)
    chord = np.asarray(chord_values, dtype=float)
    delta_x_front = np.asarray(front_values, dtype=float)
    freestream_speed = float(np.linalg.norm(solver.current_operating_point.vInf_GP1__E))
    if not np.isfinite(freestream_speed) or freestream_speed <= 0.0:
        raise RuntimeError("Hirato Eq. 6 requires nonzero Ptera freestream speed")
    scale = lesp_eq6(
        np.ones(leading_panel_indices.size),
        freestream_speed,
        chord,
        delta_x_front,
    )
    operator = np.zeros((leading_panel_indices.size, panel_count), dtype=float)
    operator[np.arange(leading_panel_indices.size), leading_panel_indices] = scale
    return PteraNativeLESPObserver(
        operator=operator,
        leading_panel_indices=leading_panel_indices,
        chord_m=chord,
        delta_x_front_m=delta_x_front,
        freestream_speed_m_s=freestream_speed,
        leading_edges_gp1_m=np.asarray(leading_edges, dtype=float),
        rear_bound_aft_edges_gp1_m=np.asarray(rear_aft_edges, dtype=float),
        chord_tangent_gp1=np.asarray(chord_tangents, dtype=float),
        suction_normal_gp1=np.asarray(suction_normals, dtype=float),
        ptera_core_radius_m=np.asarray(source_core_radii, dtype=float),
        strip_airplane_index=np.asarray(airplane_indices, dtype=int),
        strip_wing_index=np.asarray(wing_indices, dtype=int),
        strip_spanwise_index=np.asarray(spanwise_indices, dtype=int),
    )


def ptera_native_unit_ring_velocity_grid(
    evaluation_points_gp1_m: np.ndarray | Sequence[Sequence[float]],
    rings_gp1_m: np.ndarray,
    core_radius_m: np.ndarray | Sequence[float],
    *,
    nu_m2_s: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate unit rings with Ptera's own expanded ring-vortex kernel.

    ``rings_gp1_m`` uses the readable order
    ``[front-left, front-right, back-right, back-left]``.  It is mapped to
    Ptera's named vertex arguments without a circulation sign conversion.
    The returned velocity has shape ``(point, ring, xyz)``.
    """

    points = _finite_float_array(
        evaluation_points_gp1_m,
        name="evaluation_points_gp1_m",
        ndim=2,
    )
    rings = _finite_float_array(rings_gp1_m, name="rings_gp1_m", ndim=3)
    if points.shape[1:] != (3,) or rings.shape[1:] != (4, 3):
        raise ValueError("points and rings must have shapes (N,3) and (M,4,3)")
    radii = _finite_float_array(core_radius_m, name="core_radius_m", ndim=1)
    if radii.shape != (rings.shape[0],) or np.any(radii < 0.0):
        raise ValueError("core_radius_m must be nonnegative with one value per ring")
    nu = float(nu_m2_s)
    if not np.isfinite(nu) or nu < 0.0:
        raise ValueError("nu_m2_s must be finite and nonnegative")
    singularity_counts = np.zeros(4, dtype=np.int64)
    velocity = _aerodynamics_functions.expanded_velocities_from_ring_vortices(
        stackP_GP1_CgP1=points,
        stackBrrvp_GP1_CgP1=rings[:, 2],
        stackFrrvp_GP1_CgP1=rings[:, 1],
        stackFlrvp_GP1_CgP1=rings[:, 0],
        stackBlrvp_GP1_CgP1=rings[:, 3],
        strengths=np.ones(rings.shape[0], dtype=float),
        r_c0s=radii,
        singularity_counts=singularity_counts,
        ages=None,
        nu=nu,
    )
    if velocity.shape != (points.shape[0], rings.shape[0], 3) or not np.all(
        np.isfinite(velocity)
    ):
        raise FloatingPointError("Ptera unit-ring kernel returned invalid velocity")
    return velocity, singularity_counts


def _ptera_native_ring_normal_influence(
    solver: UVPMHybridSolver,
    rings_gp1_m: np.ndarray,
    core_radius_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    direct, singularity_counts = ptera_native_unit_ring_velocity_grid(
        solver.stackCpp_GP1_CgP1,
        rings_gp1_m,
        core_radius_m,
        nu_m2_s=float(solver.current_operating_point.nu),
    )
    velocity = direct
    reflection = solver.current_operating_point.surfaceReflect_T_act_GP1_CgP1
    if reflection is not None:
        reflected_points = _transformations.apply_T_to_vectors(
            reflection,
            solver.stackCpp_GP1_CgP1,
            has_point=True,
        )
        image_velocity, image_counts = ptera_native_unit_ring_velocity_grid(
            reflected_points,
            rings_gp1_m,
            core_radius_m,
            nu_m2_s=float(solver.current_operating_point.nu),
        )
        velocity = velocity + _transformations.apply_T_to_vectors(
            reflection,
            image_velocity,
            has_point=False,
        )
        singularity_counts = singularity_counts + image_counts
    normal_influence = np.einsum(
        "ijk,ik->ij",
        velocity,
        np.asarray(solver.stackUnitNormals_GP1, dtype=float),
    )
    if not np.all(np.isfinite(normal_influence)):
        raise FloatingPointError("Ptera unit-ring normal influence is non-finite")
    return normal_influence, singularity_counts


def build_ptera_native_birth_influence(
    solver: UVPMHybridSolver,
    observer: PteraNativeLESPObserver,
    a0_parent: np.ndarray | Sequence[float],
    active: np.ndarray | Sequence[bool],
) -> PteraNativeBirthInfluence:
    """Construct nascent-LEV plus pseudovortex ``B`` with Ptera's kernel."""

    a0 = _finite_float_array(a0_parent, name="a0_parent", ndim=1)
    mask = np.asarray(active, dtype=bool)
    strip_count = observer.operator.shape[0]
    if a0.shape != (strip_count,) or mask.shape != (strip_count,):
        raise ValueError("a0_parent and active must have one value per strip")
    active_indices = np.flatnonzero(mask)
    if active_indices.size == 0:
        empty_rings = np.empty((0, 4, 3), dtype=float)
        empty_influence = np.empty((observer.operator.shape[1], 0), dtype=float)
        return PteraNativeBirthInfluence(
            active_strip_indices=active_indices,
            newborn_rings_gp1_m=empty_rings,
            pseudovortex_rings_gp1_m=empty_rings.copy(),
            newborn_unit_normal_influence=empty_influence,
            pseudovortex_unit_normal_influence=empty_influence.copy(),
            combined_unit_normal_influence=empty_influence.copy(),
            singularity_counts=np.zeros(4, dtype=np.int64),
        )

    local_alpha = np.arctan2(
        -observer.chord_tangent_gp1[:, 2],
        observer.chord_tangent_gp1[:, 0],
    )
    displacement_2d = first_lev_displacement_ramesh_2d(
        observer.freestream_speed_m_s,
        a0,
        local_alpha,
        float(solver.delta_time),
    )
    displacement = embed_chord_normal_displacement(
        displacement_2d,
        observer.chord_tangent_gp1,
        observer.suction_normal_gp1,
    )
    leading = observer.leading_edges_gp1_m[active_indices]
    aft = leading + displacement[active_indices, None, :]
    newborn_rings_forward = np.stack(
        (leading[:, 0], leading[:, 1], aft[:, 1], aft[:, 0]),
        axis=1,
    )
    # Hirato's pseudovortex is superposed on the bound circulation and must
    # keep Ptera's bound-ring orientation so that its rear filament supplies
    # the +Gamma_L term in Eq. 9.  The material newborn therefore carries the
    # opposite orientation at their shared leading edge.  This gauge keeps
    # the augmented unknown itself in the same sign convention as Eq. 9.
    newborn_rings = newborn_rings_forward[:, [1, 0, 3, 2]]
    rear = observer.rear_bound_aft_edges_gp1_m[active_indices]
    pseudovortex_rings = np.stack(
        (leading[:, 0], leading[:, 1], rear[:, 0], rear[:, 1]),
        axis=1,
    )
    radii = observer.ptera_core_radius_m[active_indices]
    newborn_influence, newborn_counts = _ptera_native_ring_normal_influence(
        solver,
        newborn_rings,
        radii,
    )
    pseudo_influence, pseudo_counts = _ptera_native_ring_normal_influence(
        solver,
        pseudovortex_rings,
        radii,
    )
    return PteraNativeBirthInfluence(
        active_strip_indices=active_indices,
        newborn_rings_gp1_m=newborn_rings,
        pseudovortex_rings_gp1_m=pseudovortex_rings,
        newborn_unit_normal_influence=newborn_influence,
        pseudovortex_unit_normal_influence=pseudo_influence,
        combined_unit_normal_influence=newborn_influence + pseudo_influence,
        singularity_counts=newborn_counts + pseudo_counts,
    )


def solve_ptera_native_augmented_system(
    ptera_bound_aic: np.ndarray | Sequence[Sequence[float]],
    ptera_rhs: np.ndarray | Sequence[float],
    *,
    lev_unit_normal_influence: np.ndarray
    | Sequence[float]
    | Sequence[Sequence[float]]
    | None = None,
    lesp_operator: np.ndarray
    | Sequence[float]
    | Sequence[Sequence[float]]
    | None = None,
    lesp_target: np.ndarray | Sequence[float] | float | None = None,
    active: bool = False,
) -> PteraNativeAugmentedSolution:
    """Solve Ptera's bound system or the exact ``[A B; H 0]`` extension.

    Parameters use Ptera's existing sign convention directly::

        A @ gamma_bound = -wake_normal_velocity - freestream_normal_velocity

    ``B`` must be the normal influence of unit-strength nascent material LEV
    rings evaluated with a Ptera-compatible kernel and orientation.  This
    helper intentionally does not construct ``A`` or ``B`` itself.  ``H`` and
    ``lesp_target`` impose ``H @ gamma_bound = lesp_target``.

    When ``active`` is false, when no LEV columns are provided, or when the
    parent solution already satisfies the target bit-for-bit, the returned
    bound circulation is the direct ``np.linalg.solve(A, rhs)`` result.  That
    explicit branch is the exact parent-reduction gate.
    """

    if not isinstance(active, (bool, np.bool_)):
        raise TypeError("active must be boolean")
    aic = _finite_float_array(ptera_bound_aic, name="ptera_bound_aic", ndim=2)
    rhs = _finite_float_array(ptera_rhs, name="ptera_rhs", ndim=1)
    if aic.shape[0] != aic.shape[1]:
        raise ValueError("ptera_bound_aic must be square")
    panel_count = aic.shape[0]
    if rhs.shape != (panel_count,):
        raise ValueError("ptera_rhs length must match ptera_bound_aic")

    try:
        parent_gamma = np.linalg.solve(aic, rhs)
    except np.linalg.LinAlgError as error:
        raise FloatingPointError("Ptera bound system is singular") from error
    parent_residual = aic @ parent_gamma - rhs
    if (
        not np.all(np.isfinite(parent_gamma))
        or not np.all(np.isfinite(parent_residual))
        or float(np.max(np.abs(parent_residual), initial=0.0)) > AUGMENTED_RESIDUAL_ATOL
    ):
        raise FloatingPointError("Ptera parent solve failed its finite residual gate")

    if lev_unit_normal_influence is None:
        lev_influence = np.empty((panel_count, 0), dtype=float)
    else:
        lev_influence = np.asarray(lev_unit_normal_influence, dtype=float)
        if lev_influence.ndim == 1:
            lev_influence = lev_influence[:, None]
        if lev_influence.ndim != 2:
            raise ValueError("lev_unit_normal_influence must be 1D or 2D")
        if lev_influence.shape[0] != panel_count:
            raise ValueError(
                "lev_unit_normal_influence rows must match ptera_bound_aic"
            )
        if not np.all(np.isfinite(lev_influence)):
            raise ValueError(
                "lev_unit_normal_influence must contain only finite values"
            )
    lev_count = lev_influence.shape[1]

    if lesp_operator is None:
        operator = np.empty((0, panel_count), dtype=float)
    else:
        operator = np.asarray(lesp_operator, dtype=float)
        if operator.ndim == 1:
            operator = operator[None, :]
        if operator.ndim != 2 or operator.shape[1] != panel_count:
            raise ValueError("lesp_operator must have one column per bound panel")
        if not np.all(np.isfinite(operator)):
            raise ValueError("lesp_operator must contain only finite values")

    if lesp_target is None:
        target = np.empty(0, dtype=float)
    else:
        target = np.asarray(lesp_target, dtype=float)
        if target.ndim == 0:
            target = target.reshape(1)
        if target.ndim != 1 or target.shape[0] != operator.shape[0]:
            raise ValueError("lesp_target length must match lesp_operator rows")
        if not np.all(np.isfinite(target)):
            raise ValueError("lesp_target must contain only finite values")

    def parent_result(reason: str) -> PteraNativeAugmentedSolution:
        gamma_lev = np.zeros(lev_count, dtype=float)
        return PteraNativeAugmentedSolution(
            gamma_bound=parent_gamma,
            gamma_lev=gamma_lev,
            no_penetration_residual=parent_residual.copy(),
            lesp_constraint_residual=operator @ parent_gamma - target,
            augmented_matrix=aic.copy(),
            augmented_rhs=rhs.copy(),
            used_augmented_system=False,
            reduction_reason=reason,
        )

    if not bool(active):
        return parent_result("inactive")
    if lev_count == 0:
        return parent_result("no_active_lev_columns")
    if operator.shape[0] != lev_count:
        raise ValueError(
            "active [A B; H 0] solve requires one LESP constraint per LEV unknown"
        )
    if np.array_equal(operator @ parent_gamma, target):
        return parent_result("parent_exactly_satisfies_constraint")

    zero_block = np.zeros((lev_count, lev_count), dtype=float)
    augmented_matrix = np.block([[aic, lev_influence], [operator, zero_block]])
    augmented_rhs = np.concatenate((rhs, target))
    try:
        augmented_solution = np.linalg.solve(augmented_matrix, augmented_rhs)
    except np.linalg.LinAlgError as error:
        raise FloatingPointError("native augmented system is singular") from error
    if not np.all(np.isfinite(augmented_solution)):
        raise FloatingPointError("native augmented solution is non-finite")
    gamma_bound = augmented_solution[:panel_count]
    gamma_lev = augmented_solution[panel_count:]
    no_penetration_residual = aic @ gamma_bound + lev_influence @ gamma_lev - rhs
    lesp_constraint_residual = operator @ gamma_bound - target
    residuals_are_finite = np.all(np.isfinite(no_penetration_residual)) and np.all(
        np.isfinite(lesp_constraint_residual)
    )
    maximum_residual = max(
        float(np.max(np.abs(no_penetration_residual), initial=0.0)),
        float(np.max(np.abs(lesp_constraint_residual), initial=0.0)),
    )
    if not residuals_are_finite or maximum_residual > AUGMENTED_RESIDUAL_ATOL:
        raise FloatingPointError("native augmented solve failed its residual gate")
    return PteraNativeAugmentedSolution(
        gamma_bound=gamma_bound,
        gamma_lev=gamma_lev,
        no_penetration_residual=no_penetration_residual,
        lesp_constraint_residual=lesp_constraint_residual,
        augmented_matrix=augmented_matrix,
        augmented_rhs=augmented_rhs,
        used_augmented_system=True,
        reduction_reason=None,
    )


def native_active_probe(
    solver: UVPMHybridSolver,
    *,
    lesp_critical: float,
    commit_bound_strengths: bool = True,
) -> NativeActiveProbeReport:
    """Propose one active LESP solve on an already parent-solved Ptera step.

    This is intentionally not a time-marching material-wake implementation.
    The caller must stop the Ptera step after this probe: no LEV history is
    committed and this function never calls Ptera's wake or load methods.
    All geometry, kernel, AIC and right-hand-side state is read from the
    current Ptera step.  The only optional mutation is a transactional update
    of the current bound-strength array and matching Ptera RingVortex objects.
    """

    threshold = float(lesp_critical)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("lesp_critical must be finite and positive")
    if not isinstance(commit_bound_strengths, (bool, np.bool_)):
        raise TypeError("commit_bound_strengths must be boolean")

    aic = _finite_float_array(
        solver._currentGridWingWingInfluences__E,
        name="_currentGridWingWingInfluences__E",
        ndim=2,
    ).copy()
    wake = _finite_float_array(
        solver._currentStackWakeWingInfluences__E,
        name="_currentStackWakeWingInfluences__E",
        ndim=1,
    ).copy()
    freestream = _finite_float_array(
        solver._currentStackFreestreamWingInfluences__E,
        name="_currentStackFreestreamWingInfluences__E",
        ndim=1,
    ).copy()
    parent_gamma = _finite_float_array(
        solver._current_bound_vortex_strengths,
        name="_current_bound_vortex_strengths",
        ndim=1,
    ).copy()
    panel_count = parent_gamma.size
    if (
        aic.shape != (panel_count, panel_count)
        or wake.shape != (panel_count,)
        or freestream.shape != (panel_count,)
    ):
        raise RuntimeError("current Ptera AIC/RHS/bound state shapes disagree")
    rhs = -wake - freestream
    parent_solution = solve_ptera_native_augmented_system(aic, rhs, active=False)
    if not np.array_equal(parent_solution.gamma_bound, parent_gamma):
        raise RuntimeError(
            "current Ptera bound state is not the direct parent AIC/RHS solve"
        )

    observer = build_ptera_native_lesp_observer(solver)
    a0_parent = observer.observe(parent_gamma)
    direct_a0_parent = observer.direct_eq6(parent_gamma)
    if not np.allclose(
        a0_parent,
        direct_a0_parent,
        rtol=2.0e-15,
        atol=2.0e-15,
    ):
        raise FloatingPointError("Ptera H observer disagrees with direct Hirato Eq. 6")
    active = preconstraint_shed_mask(a0_parent, threshold)
    birth = build_ptera_native_birth_influence(
        solver,
        observer,
        a0_parent,
        active,
    )

    strip_count = active.size
    gamma_lev_by_strip = np.zeros(strip_count, dtype=float)
    if np.any(active):
        active_operator = observer.operator[active]
        target = threshold * np.sign(a0_parent[active])
        augmented_solution = solve_ptera_native_augmented_system(
            aic,
            rhs,
            lev_unit_normal_influence=birth.combined_unit_normal_influence,
            lesp_operator=active_operator,
            lesp_target=target,
            active=True,
        )
        gamma_lev_by_strip[active] = augmented_solution.gamma_lev
    else:
        augmented_solution = parent_solution
    proposed_gamma = augmented_solution.gamma_bound.copy()
    a0_post = observer.observe(proposed_gamma)
    if np.any(active):
        active_residual = a0_post[active] - (threshold * np.sign(a0_parent[active]))
        if (
            not np.all(np.isfinite(active_residual))
            or float(np.max(np.abs(active_residual), initial=0.0))
            > AUGMENTED_RESIDUAL_ATOL
        ):
            raise FloatingPointError("native active probe failed its Eq. 6 gate")
    elif not np.array_equal(proposed_gamma, parent_gamma):
        raise FloatingPointError("inactive native probe did not reduce to parent")

    panels = list(np.ravel(solver.panels))
    if len(panels) != panel_count:
        raise RuntimeError("Ptera panel list changed during native proposal")
    old_ring_strengths = np.empty(panel_count, dtype=float)
    for panel_index, panel in enumerate(panels):
        if panel.ring_vortex is None:
            raise RuntimeError("Ptera panel is missing its bound RingVortex")
        old_ring_strengths[panel_index] = float(panel.ring_vortex.strength)
    if not np.array_equal(old_ring_strengths, parent_gamma):
        raise RuntimeError("Ptera RingVortex objects disagree with bound state")

    committed = bool(commit_bound_strengths and np.any(active))
    if committed:
        old_bound_array = solver._current_bound_vortex_strengths
        try:
            solver._current_bound_vortex_strengths = proposed_gamma.copy()
            for panel, strength in zip(panels, proposed_gamma, strict=True):
                panel.ring_vortex.strength = float(strength)
        except Exception:
            solver._current_bound_vortex_strengths = old_bound_array
            for panel, strength in zip(panels, old_ring_strengths, strict=True):
                panel.ring_vortex.strength = float(strength)
            raise

    return NativeActiveProbeReport(
        lesp_critical=threshold,
        parent_gamma_bound=parent_gamma,
        proposed_gamma_bound=proposed_gamma,
        gamma_lev_by_strip=gamma_lev_by_strip,
        a0_parent=a0_parent,
        a0_post=a0_post,
        active=active,
        observer=observer,
        birth_influence=birth,
        augmented_solution=augmented_solution,
        committed_bound_strengths=committed,
        force_scoring_status=FORCE_SCORING_STATUS,
        state_scope=(
            "single_parent_solved_step_bound_only_no_material_history_no_wake_no_load"
        ),
    )


class NativeMaterialLEVUVPMHybridSolver(UVPMHybridSolver):
    """Passive Stage-1 subclass with a pristine material-LEV history.

    Construction is restricted to an unreachable LESP threshold.  Every
    aerodynamic override immediately invokes the same method on ``super()`` and
    returns its result unchanged.  The overrides make the reduction contract
    explicit and give future material-wake work narrow integration points.
    """

    force_scoring_status: Final[str] = FORCE_SCORING_STATUS

    def __init__(
        self,
        unsteady_problem: Any,
        *,
        v5f_config: NativeMaterialLEVConfig,
        **uvpm_kwargs: Any,
    ) -> None:
        if not v5f_config.enabled:
            raise ValueError(
                "disabled v5f configurations must use make_fluxv_v5f_solver"
            )
        if not np.isposinf(float(v5f_config.lesp_critical)):
            raise NotImplementedError(
                "active material LEV is blocked until the material wake and "
                "Ptera load corrections are implemented"
            )
        self.v5f_config = v5f_config
        self.material_lev_history: tuple[Any, ...] = ()
        super().__init__(unsteady_problem, **uvpm_kwargs)

    def _calculate_wing_wing_influences(self) -> None:
        return super()._calculate_wing_wing_influences()

    def _calculate_wake_wing_influences(self) -> None:
        return super()._calculate_wake_wing_influences()

    def _calculate_vortex_strengths(self) -> None:
        return super()._calculate_vortex_strengths()

    def calculate_solution_velocity(
        self,
        stackP_GP1_CgP1: np.ndarray | Sequence[Sequence[float | int]],
        bound_singularity_counts: np.ndarray | None = None,
        wake_singularity_counts: np.ndarray | None = None,
    ) -> np.ndarray:
        return super().calculate_solution_velocity(
            stackP_GP1_CgP1,
            bound_singularity_counts=bound_singularity_counts,
            wake_singularity_counts=wake_singularity_counts,
        )

    def _calculate_loads(self) -> None:
        return super()._calculate_loads()

    def _populate_next_airplanes_wake(self) -> None:
        return super()._populate_next_airplanes_wake()


def make_fluxv_v5f_solver(
    unsteady_problem: Any,
    *,
    config: NativeMaterialLEVConfig = DEFAULT_NATIVE_MATERIAL_LEV_CONFIG,
    **uvpm_kwargs: Any,
) -> UVPMHybridSolver:
    """Build the direct parent solver or the passive v5f Stage-1 subclass."""

    if not isinstance(config, NativeMaterialLEVConfig):
        raise TypeError("config must be a NativeMaterialLEVConfig")
    if not bool(config.enabled):
        return UVPMHybridSolver(unsteady_problem, **uvpm_kwargs)
    return NativeMaterialLEVUVPMHybridSolver(
        unsteady_problem,
        v5f_config=config,
        **uvpm_kwargs,
    )


__all__ = [
    "AUGMENTED_RESIDUAL_ATOL",
    "DEFAULT_NATIVE_MATERIAL_LEV_CONFIG",
    "FORCE_SCORING_STATUS",
    "NativeActiveProbeReport",
    "NativeMaterialLEVConfig",
    "NativeMaterialLEVUVPMHybridSolver",
    "PteraNativeBirthInfluence",
    "PteraNativeLESPObserver",
    "PteraNativeAugmentedSolution",
    "build_ptera_native_birth_influence",
    "build_ptera_native_lesp_observer",
    "make_fluxv_v5f_solver",
    "native_active_probe",
    "ptera_native_unit_ring_velocity_grid",
    "solve_ptera_native_augmented_system",
]
