"""Read-only Ptera trailing-edge wake to rVPM diagnostic-shadow adapter.

This is the minimal G4 adapter for the v5h R2/B2 experiment.  It extracts the
front row of one Ptera wake from the wing's shared vertex grid, verifies the
matching ``RingVortex`` object strength/age time layer, and delegates topology
and conservative deposition to :mod:`fluxvortex.rvpm_edge_bridge`.

The adapter is intentionally one-way.  Ptera rings remain the physical owner;
the returned particles never enter Ptera's AIC, wake update, or load path.
Only one instantaneous Ptera kernel channel is accepted per call, so edges
with heterogeneous core/age state can never be merged accidentally.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from numbers import Integral, Real
from typing import Any, Literal

import numpy as np

from fluxvortex.rvpm_edge_bridge import (
    DIAGNOSTIC_SHADOW_OWNER,
    FROZEN_OVERLAP_LAMBDA,
    RING_PHYSICAL_OWNER,
    BridgeNode,
    DirectedRing,
    ShadowBridgeResult,
    build_diagnostic_shadow_bridge,
)

PTERA_SQUIRE_PARAMETER = 1.0e-4
PTERA_LAMB_CONSTANT = 1.25643
PTERA_INITIAL_CORE_FRACTION = 0.03
TE_WAKE_ROW = 0

TimeLayer = Literal["current", "next"]


@dataclass(frozen=True)
class PteraTEKernelProvenance:
    """Exact source/shadow kernel channels for one TE snapshot."""

    source_channel_id: str
    source_kernel_family: str
    source_core_law: str
    source_initial_core_radius_m: float
    source_age_s: float
    source_kinematic_viscosity_m2_s: float
    source_effective_core_radius_m: float
    shadow_channel_id: str
    shadow_kernel_family: str
    shadow_sigma_law: str
    topology_merge_scope: str
    source_shadow_merge_allowed: bool


@dataclass(frozen=True)
class PteraTEShadowDiagnostics:
    """Read-only extraction and ownership gates for one adapter call."""

    enabled: bool
    time_layer: TimeLayer | None
    source_step: int | None
    airplane_index: int | None
    wing_index: int | None
    wake_row: int | None
    source_ring_count: int
    source_node_count: int
    strength_fact_source: str | None
    strength_layer_verified: bool
    ring_object_geometry_verified: bool
    prescribed_wake: bool | None
    physical_owner: str
    shadow_owner: str
    source_kernel_channel_count: int
    heterogeneous_kernel_merge_count: int
    source_shadow_merge_count: int
    parent_write_count: int
    load_write_count: int
    feedback_call_count: int


@dataclass(frozen=True)
class PteraTEShadowResult:
    """Auditable Ptera source snapshot and conservative shadow particles."""

    bridge: ShadowBridgeResult
    source_ring_ids: tuple[str, ...]
    source_ring_node_ids: tuple[tuple[str, str, str, str], ...]
    source_strengths_m2_s: tuple[float, ...]
    source_ages_s: tuple[float, ...]
    provenance: PteraTEKernelProvenance | None
    diagnostics: PteraTEShadowDiagnostics


def _integer(name: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return result


def _finite_real(name: str, value: object, *, nonnegative: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    if nonnegative and result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _select(sequence: object, index: int, *, name: str) -> Any:
    try:
        length = len(sequence)  # type: ignore[arg-type]
    except TypeError as error:
        raise RuntimeError(f"Ptera {name} is not an indexable sequence") from error
    if index >= length:
        raise ValueError(f"{name} index {index} is outside [0, {length})")
    return sequence[index]  # type: ignore[index]


def _numeric_array(name: str, value: object, *, ndim: int) -> np.ndarray:
    original = np.asarray(value)
    if original.dtype.kind not in "iuf" or original.dtype.kind == "b":
        raise RuntimeError(f"Ptera {name} must use a real numeric dtype")
    array = np.asarray(original, dtype=np.float64)
    if array.ndim != ndim or not np.all(np.isfinite(array)):
        raise RuntimeError(f"Ptera {name} has an invalid shape or nonfinite values")
    return array


def _wake_object_arrays(
    airplanes: object,
) -> tuple[np.ndarray, np.ndarray, dict[tuple[int, int], slice]]:
    strengths: list[float] = []
    ages: list[float] = []
    wing_slices: dict[tuple[int, int], slice] = {}
    start = 0
    for airplane_index, airplane in enumerate(airplanes):  # type: ignore[arg-type]
        wings = getattr(airplane, "wings", None)
        if wings is None:
            raise RuntimeError("Ptera airplane is missing wings")
        for wing_index, wing in enumerate(wings):
            wake = np.asarray(getattr(wing, "wake_ring_vortices", None), dtype=object)
            if wake.ndim != 2:
                raise RuntimeError("Ptera wake_ring_vortices must be a 2D object grid")
            for ring in wake.ravel():
                strengths.append(
                    _finite_real("RingVortex.strength", getattr(ring, "strength", None))
                )
                ages.append(
                    _finite_real(
                        "RingVortex.age",
                        getattr(ring, "age", None),
                        nonnegative=True,
                    )
                )
            stop = start + wake.size
            wing_slices[(airplane_index, wing_index)] = slice(start, stop)
            start = stop
    return (
        np.asarray(strengths, dtype=np.float64),
        np.asarray(ages, dtype=np.float64),
        wing_slices,
    )


def _verify_current_fact_source(
    solver: object,
    *,
    airplanes: object,
) -> str:
    object_strengths, object_ages, _ = _wake_object_arrays(airplanes)
    collapsed_strengths = _numeric_array(
        "_current_wake_vortex_strengths",
        getattr(solver, "_current_wake_vortex_strengths", None),
        ndim=1,
    )
    collapsed_ages = _numeric_array(
        "_current_wake_vortex_ages",
        getattr(solver, "_current_wake_vortex_ages", None),
        ndim=1,
    )
    if not np.array_equal(collapsed_strengths, object_strengths):
        raise RuntimeError(
            "current RingVortex object strengths disagree with the collapsed "
            "current strength time layer"
        )
    if not np.array_equal(collapsed_ages, object_ages):
        raise RuntimeError(
            "current RingVortex object ages disagree with the collapsed current "
            "age time layer"
        )
    return (
        "current wing.wake_ring_vortices verified against post-collapse current arrays"
    )


def _front_bound_strengths(wing: object) -> np.ndarray:
    panels = np.asarray(getattr(wing, "panels", None), dtype=object)
    if panels.ndim != 2 or panels.shape[0] == 0 or panels.shape[1] == 0:
        raise RuntimeError("Ptera current wing panels have an invalid topology")
    strengths: list[float] = []
    for panel in panels[-1, :]:
        ring = getattr(panel, "ring_vortex", None)
        if ring is None:
            raise RuntimeError("Ptera trailing-edge panel is missing its RingVortex")
        strengths.append(
            _finite_real("bound RingVortex.strength", getattr(ring, "strength", None))
        )
    return np.asarray(strengths, dtype=np.float64)


def _ring_scalar_grid(wing: object, attribute: str) -> np.ndarray:
    wake = np.asarray(getattr(wing, "wake_ring_vortices", None), dtype=object)
    if wake.ndim != 2:
        raise RuntimeError("Ptera wake_ring_vortices must be a 2D object grid")
    values = np.empty(wake.shape, dtype=np.float64)
    for index, ring in np.ndenumerate(wake):
        values[index] = _finite_real(
            f"RingVortex.{attribute}",
            getattr(ring, attribute, None),
            nonnegative=attribute == "age",
        )
    return values


def _verify_next_fact_source(
    solver: object,
    *,
    current_wing: object,
    next_wing: object,
) -> str:
    next_strengths = _ring_scalar_grid(next_wing, "strength")
    next_ages = _ring_scalar_grid(next_wing, "age")
    current_wake = np.asarray(
        getattr(current_wing, "wake_ring_vortices", None), dtype=object
    )
    if current_wake.ndim != 2:
        raise RuntimeError("Ptera current wake_ring_vortices is not a 2D grid")

    max_rows = getattr(solver, "_max_wake_rows", None)
    retained_current = current_wake
    if max_rows is not None:
        maximum = _integer("_max_wake_rows", max_rows, minimum=1)
        if current_wake.shape[0] >= maximum:
            retained_current = current_wake[: maximum - 1]

    expected_shape = (retained_current.shape[0] + 1, current_wake.shape[1])
    if next_strengths.shape != expected_shape:
        raise RuntimeError(
            "next RingVortex object grid has the wrong current-to-next time layer shape"
        )
    expected_strengths = np.vstack(
        (
            _front_bound_strengths(current_wing)[None, :],
            _ring_scalar_grid_from_objects(retained_current, "strength"),
        )
    )
    if not np.array_equal(next_strengths, expected_strengths):
        raise RuntimeError(
            "next RingVortex object strengths do not equal current TE bound plus "
            "retained current-wake strengths"
        )

    expected_ages = np.zeros(expected_shape, dtype=np.float64)
    if retained_current.shape[0]:
        current_step = _integer("_current_step", getattr(solver, "_current_step", None))
        delta_time = _finite_real(
            "delta_time", getattr(solver, "delta_time", None), nonnegative=True
        )
        old_ages = _ring_scalar_grid_from_objects(retained_current, "age")
        expected_ages[1:] = delta_time if current_step == 0 else old_ages + delta_time
    if not np.array_equal(next_ages, expected_ages):
        raise RuntimeError(
            "next RingVortex object ages do not match Ptera's current-to-next age law"
        )
    return "next wing.wake_ring_vortices pre-collapse object fact source"


def _ring_scalar_grid_from_objects(wake: np.ndarray, attribute: str) -> np.ndarray:
    values = np.empty(wake.shape, dtype=np.float64)
    for index, ring in np.ndenumerate(wake):
        values[index] = _finite_real(
            f"RingVortex.{attribute}",
            getattr(ring, attribute, None),
            nonnegative=attribute == "age",
        )
    return values


def _verify_ring_geometry(
    wing: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid = _numeric_array(
        "wing.gridWrvp_GP1_CgP1",
        getattr(wing, "gridWrvp_GP1_CgP1", None),
        ndim=3,
    )
    if grid.shape[2] != 3:
        raise RuntimeError("Ptera wake vertex grid must have shape (rows, span, 3)")
    wake = np.asarray(getattr(wing, "wake_ring_vortices", None), dtype=object)
    if wake.ndim != 2 or wake.shape[0] < 1:
        raise RuntimeError("Ptera TE shadow requires at least one wake RingVortex row")
    if grid.shape[:2] != (wake.shape[0] + 1, wake.shape[1] + 1):
        raise RuntimeError("Ptera wake object and shared-vertex grid shapes disagree")

    strengths = _ring_scalar_grid(wing, "strength")[TE_WAKE_ROW].copy()
    ages = _ring_scalar_grid(wing, "age")[TE_WAKE_ROW].copy()
    for span_index, ring in enumerate(wake[TE_WAKE_ROW]):
        expected = (
            grid[TE_WAKE_ROW + 1, span_index + 1],  # BR
            grid[TE_WAKE_ROW, span_index + 1],  # FR
            grid[TE_WAKE_ROW, span_index],  # FL
            grid[TE_WAKE_ROW + 1, span_index],  # BL
        )
        actual = (
            getattr(ring, "Brrvp_GP1_CgP1", None),
            getattr(ring, "Frrvp_GP1_CgP1", None),
            getattr(ring, "Flrvp_GP1_CgP1", None),
            getattr(ring, "Blrvp_GP1_CgP1", None),
        )
        for corner_name, actual_corner, expected_corner in zip(
            ("BR", "FR", "FL", "BL"), actual, expected, strict=True
        ):
            corner = _numeric_array(f"RingVortex.{corner_name}", actual_corner, ndim=1)
            if corner.shape != (3,) or not np.array_equal(corner, expected_corner):
                raise RuntimeError(
                    "Ptera RingVortex corner objects disagree with the shared "
                    "wake vertex grid"
                )
    return grid, strengths, ages


def _kernel_provenance(
    *,
    wing: object,
    strengths: np.ndarray,
    ages: np.ndarray,
    kinematic_viscosity: object,
) -> tuple[PteraTEKernelProvenance, str]:
    mean_chord = _finite_real(
        "wing.standard_mean_chord",
        getattr(wing, "standard_mean_chord", None),
    )
    if mean_chord <= 0.0:
        raise RuntimeError("Ptera wing.standard_mean_chord must be positive")
    nu = _finite_real("operating_point.nu", kinematic_viscosity, nonnegative=True)
    initial_core = PTERA_INITIAL_CORE_FRACTION * mean_chord
    effective_squared = (
        initial_core**2
        + 4.0
        * PTERA_LAMB_CONSTANT
        * (nu + PTERA_SQUIRE_PARAMETER * np.abs(strengths))
        * ages
    )
    if not np.all(np.isfinite(effective_squared)) or np.any(effective_squared <= 0.0):
        raise RuntimeError("Ptera TE source produced an invalid effective core radius")
    effective = np.sqrt(effective_squared)

    # The pure edge graph algebraically combines signed incidences.  That is
    # valid only when every contributing finite segment uses the exact same
    # instantaneous regularization channel.  Reject rather than silently merge
    # heterogeneous Ptera kernels.
    if not np.all(ages == ages[0]) or not np.all(effective == effective[0]):
        raise RuntimeError(
            "heterogeneous Ptera kernel channels must be adapted separately; "
            "shared edges were not merged"
        )

    source_signature = "|".join(
        (
            "ptera-modified-lamb-oseen-finite-segment-v1",
            f"rc0={initial_core.hex()}",
            f"age={float(ages[0]).hex()}",
            f"nu={nu.hex()}",
            f"effective_rc={float(effective[0]).hex()}",
            f"squire={PTERA_SQUIRE_PARAMETER.hex()}",
            f"lamb={PTERA_LAMB_CONSTANT.hex()}",
        )
    )
    channel_token = sha256(source_signature.encode("ascii")).hexdigest()[:20]
    provenance = PteraTEKernelProvenance(
        source_channel_id=source_signature,
        source_kernel_family="ptera_modified_lamb_oseen_finite_segment",
        source_core_law=("rc^2=rc0^2+4*1.25643*(nu+1e-4*abs(gamma))*age"),
        source_initial_core_radius_m=initial_core,
        source_age_s=float(ages[0]),
        source_kinematic_viscosity_m2_s=nu,
        source_effective_core_radius_m=float(effective[0]),
        shadow_channel_id=(
            f"flowvpm_gaussian_erf_direct:sigma={FROZEN_OVERLAP_LAMBDA.hex()}*h"
        ),
        shadow_kernel_family="flowvpm_gaussian_erf_direct",
        shadow_sigma_law=f"sigma={FROZEN_OVERLAP_LAMBDA}*subsegment_length",
        topology_merge_scope=(
            "single_wing_front_te_row_identical_instantaneous_ptera_kernel_only"
        ),
        source_shadow_merge_allowed=False,
    )
    return provenance, channel_token


def _layer_state(
    solver: object,
    *,
    time_layer: object,
    airplane_index: int,
    wing_index: int,
) -> tuple[int, object, object, str]:
    if not isinstance(time_layer, str) or time_layer not in ("current", "next"):
        raise ValueError("time_layer must be 'current' or 'next'")
    current_step = _integer("_current_step", getattr(solver, "_current_step", None))
    steady_problems = getattr(solver, "steady_problems", None)
    if steady_problems is None:
        raise RuntimeError("solver is missing Ptera steady_problems")

    current_airplanes = getattr(solver, "current_airplanes", None)
    current_airplane = _select(
        current_airplanes, airplane_index, name="current airplane"
    )
    current_wing = _select(
        getattr(current_airplane, "wings", None), wing_index, name="current wing"
    )

    if time_layer == "current":
        problem = _select(steady_problems, current_step, name="steady problem")
        airplane = _select(
            getattr(problem, "airplanes", None), airplane_index, name="airplane"
        )
        wing = _select(getattr(airplane, "wings", None), wing_index, name="wing")
        if wing is not current_wing:
            raise RuntimeError("Ptera current_airplanes does not match current step")
        fact_source = _verify_current_fact_source(solver, airplanes=current_airplanes)
        return current_step, problem, wing, fact_source

    next_step = current_step + 1
    num_steps = _integer("num_steps", getattr(solver, "num_steps", None), minimum=1)
    if next_step >= num_steps:
        raise ValueError("next time layer does not exist at the final Ptera step")
    problem = _select(steady_problems, next_step, name="next steady problem")
    airplane = _select(
        getattr(problem, "airplanes", None), airplane_index, name="next airplane"
    )
    wing = _select(getattr(airplane, "wings", None), wing_index, name="next wing")
    fact_source = _verify_next_fact_source(
        solver, current_wing=current_wing, next_wing=wing
    )
    return next_step, problem, wing, fact_source


def build_ptera_te_diagnostic_shadow(
    solver: object,
    *,
    time_layer: TimeLayer | object,
    airplane_index: int | object = 0,
    wing_index: int | object = 0,
    subdivisions: int | object,
    enabled: bool = True,
) -> PteraTEShadowResult:
    """Build a conservative read-only shadow of one Ptera TE wake row.

    ``current`` validates RingVortex objects against Ptera's post-collapse
    current arrays.  ``next`` deliberately uses the newly populated RingVortex
    objects and validates them against current TE/current-wake lineage; Ptera's
    preallocated next flat arrays are not read before the next collapse.

    The disabled branch is exact and input blind: no solver, layer, index, or
    deposition argument is inspected.
    """

    if not isinstance(enabled, (bool, np.bool_)):
        raise ValueError("enabled must be boolean")
    if not bool(enabled):
        bridge = build_diagnostic_shadow_bridge(
            solver,
            solver,
            subdivisions=subdivisions,
            step=time_layer,
            enabled=False,
            physical_owner=airplane_index,
            owner_state=wing_index,
            overlap_lambda=solver,
        )
        return PteraTEShadowResult(
            bridge=bridge,
            source_ring_ids=(),
            source_ring_node_ids=(),
            source_strengths_m2_s=(),
            source_ages_s=(),
            provenance=None,
            diagnostics=PteraTEShadowDiagnostics(
                enabled=False,
                time_layer=None,
                source_step=None,
                airplane_index=None,
                wing_index=None,
                wake_row=None,
                source_ring_count=0,
                source_node_count=0,
                strength_fact_source=None,
                strength_layer_verified=False,
                ring_object_geometry_verified=False,
                prescribed_wake=None,
                physical_owner=RING_PHYSICAL_OWNER,
                shadow_owner=DIAGNOSTIC_SHADOW_OWNER,
                source_kernel_channel_count=0,
                heterogeneous_kernel_merge_count=0,
                source_shadow_merge_count=0,
                parent_write_count=0,
                load_write_count=0,
                feedback_call_count=0,
            ),
        )

    airplane_id = _integer("airplane_index", airplane_index)
    wing_id = _integer("wing_index", wing_index)
    subdivision_count = _integer("subdivisions", subdivisions, minimum=1)
    prescribed_wake = getattr(solver, "_prescribed_wake", None)
    if not isinstance(prescribed_wake, (bool, np.bool_)) or not bool(prescribed_wake):
        raise RuntimeError(
            "v5h G4 is restricted to the current FluxV prescribed Ptera wake"
        )

    source_step, problem, wing, fact_source = _layer_state(
        solver,
        time_layer=time_layer,
        airplane_index=airplane_id,
        wing_index=wing_id,
    )
    grid, strengths, ages = _verify_ring_geometry(wing)
    operating_point = getattr(problem, "operating_point", None)
    if operating_point is None:
        raise RuntimeError("Ptera steady problem is missing its operating point")
    provenance, channel_token = _kernel_provenance(
        wing=wing,
        strengths=strengths,
        ages=ages,
        kinematic_viscosity=getattr(operating_point, "nu", None),
    )

    prefix = f"ptera-te:{channel_token}:t{source_step}:a{airplane_id}:w{wing_id}"
    nodes: list[BridgeNode] = []
    for row_index in (TE_WAKE_ROW, TE_WAKE_ROW + 1):
        for span_node_index in range(grid.shape[1]):
            nodes.append(
                BridgeNode(
                    node_id=f"{prefix}:node:r{row_index}:s{span_node_index}",
                    position=tuple(
                        float(value) for value in grid[row_index, span_node_index]
                    ),
                )
            )

    ring_ids: list[str] = []
    ring_node_ids: list[tuple[str, str, str, str]] = []
    rings: list[DirectedRing] = []
    for span_index, circulation in enumerate(strengths):
        ring_id = f"{prefix}:ring:r{TE_WAKE_ROW}:s{span_index}"
        # Ptera's positive ring traversal is BR -> FR -> FL -> BL -> BR.
        node_ids = (
            f"{prefix}:node:r{TE_WAKE_ROW + 1}:s{span_index + 1}",
            f"{prefix}:node:r{TE_WAKE_ROW}:s{span_index + 1}",
            f"{prefix}:node:r{TE_WAKE_ROW}:s{span_index}",
            f"{prefix}:node:r{TE_WAKE_ROW + 1}:s{span_index}",
        )
        ring_ids.append(ring_id)
        ring_node_ids.append(node_ids)
        rings.append(
            DirectedRing(
                ring_id=ring_id,
                node_ids=node_ids,
                circulation=float(circulation),
            )
        )

    bridge = build_diagnostic_shadow_bridge(
        nodes,
        rings,
        subdivisions=subdivision_count,
        step=source_step,
        enabled=True,
        physical_owner=RING_PHYSICAL_OWNER,
        owner_state=DIAGNOSTIC_SHADOW_OWNER,
        overlap_lambda=FROZEN_OVERLAP_LAMBDA,
    )
    if bridge.feedback_velocity is not None or bridge.diagnostics.feedback_call_count:
        raise AssertionError("diagnostic Ptera TE shadow unexpectedly gained feedback")

    return PteraTEShadowResult(
        bridge=bridge,
        source_ring_ids=tuple(ring_ids),
        source_ring_node_ids=tuple(ring_node_ids),
        source_strengths_m2_s=tuple(float(value) for value in strengths),
        source_ages_s=tuple(float(value) for value in ages),
        provenance=provenance,
        diagnostics=PteraTEShadowDiagnostics(
            enabled=True,
            time_layer=time_layer,  # type: ignore[arg-type]
            source_step=source_step,
            airplane_index=airplane_id,
            wing_index=wing_id,
            wake_row=TE_WAKE_ROW,
            source_ring_count=len(rings),
            source_node_count=len(nodes),
            strength_fact_source=fact_source,
            strength_layer_verified=True,
            ring_object_geometry_verified=True,
            prescribed_wake=True,
            physical_owner=RING_PHYSICAL_OWNER,
            shadow_owner=DIAGNOSTIC_SHADOW_OWNER,
            source_kernel_channel_count=1,
            heterogeneous_kernel_merge_count=0,
            source_shadow_merge_count=0,
            parent_write_count=0,
            load_write_count=0,
            feedback_call_count=bridge.diagnostics.feedback_call_count,
        ),
    )


__all__ = [
    "PTERA_INITIAL_CORE_FRACTION",
    "PTERA_LAMB_CONSTANT",
    "PTERA_SQUIRE_PARAMETER",
    "PteraTEKernelProvenance",
    "PteraTEShadowDiagnostics",
    "PteraTEShadowResult",
    "TE_WAKE_ROW",
    "build_ptera_te_diagnostic_shadow",
]
