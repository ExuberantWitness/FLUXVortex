"""Causal incremental connected-sheet oracle for FluxV v5h8.

This module is deliberately a manufactured, observation-free mechanical
oracle.  It distinguishes an append-only *contribution view* from a derived
canonical net view.  Once a sheet boundary has been transported, the new
upstream incidence is represented by a fresh particle layer that clones the
old live boundary's support and scales its vector strength.  The old particle
prefix, IDs, and birth lineage are never rewritten.

The construction is not a production remesher and has no aerodynamic force,
Ptera, feedback, or target-data path.  Fresh geometry redeposition is exposed
only as a negative-control comparator.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum
from numbers import Integral, Real
from typing import Any, Literal, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from fluxvortex.rvpm_edge_bridge import (
    BridgeNode,
    DIAGNOSTIC_SHADOW_OWNER,
    DirectedRing,
    EdgeKey,
    ParticleLineage,
    RING_PHYSICAL_OWNER,
    assemble_ring_edge_graph,
    canonical_edge_key,
    deposit_edge_graph_prescribed_sigma_and_spacing,
)
from fluxvortex.rvpm_reference import direct_gaussian_erf_velocity_jacobian


FloatArray = NDArray[np.float64]
ParticleRole = Literal[
    "fresh_upstream",
    "fresh_downstream",
    "fresh_left_tip",
    "fresh_right_tip",
    "inherited_upstream_counter",
]

INTERFACE_ID = "fluxv-v5h8-incremental-connected-sheet-v1"
CLONE_PARTICLE_SCHEMA = "fluxv-v5h8-live-basis-clone-v1"


def _finite_real(name: str, value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _positive_real(name: str, value: object) -> float:
    result = _finite_real(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be strictly positive")
    return result


def _positive_integer(name: str, value: object) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _point(name: str, value: ArrayLike) -> tuple[float, float, float]:
    original = np.asarray(value)
    if original.dtype.kind not in "iuf" or original.dtype.kind == "b":
        raise TypeError(f"{name} must use a real numeric dtype")
    array = np.asarray(original, dtype=np.float64)
    if array.shape != (3,):
        raise ValueError(f"{name} must have shape (3,)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return (float(array[0]), float(array[1]), float(array[2]))


def _matrix(name: str, value: ArrayLike) -> FloatArray:
    original = np.asarray(value)
    if original.dtype.kind not in "iuf" or original.dtype.kind == "b":
        raise TypeError(f"{name} must use a real numeric dtype")
    array = np.asarray(original, dtype=np.float64)
    if array.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    determinant = float(np.linalg.det(array))
    if not np.isfinite(determinant) or abs(determinant) <= np.finfo(np.float64).tiny:
        raise ValueError(f"{name} must be nonsingular")
    return np.ascontiguousarray(array)


def _readonly_array(name: str, value: ArrayLike, *, ndim: int) -> FloatArray:
    original = np.asarray(value)
    if original.dtype.kind not in "iuf" or original.dtype.kind == "b":
        raise TypeError(f"{name} must use a real numeric dtype")
    array = np.ascontiguousarray(np.asarray(original, dtype=np.float64))
    if array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    # A bytes-backed view cannot be made writeable again with ``setflags``.
    # This makes the public frozen state materially immutable rather than only
    # carrying a reversible NumPy write flag.
    immutable = np.frombuffer(array.tobytes(order="C"), dtype=np.float64).reshape(
        array.shape
    )
    immutable.setflags(write=False)
    return immutable


def _validate_readonly_particle_arrays(
    positions: object,
    gamma: object,
    sigma: object,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    arrays = (positions, gamma, sigma)
    names = ("positions", "gamma", "sigma")
    for name, array in zip(names, arrays, strict=True):
        if type(array) is not np.ndarray:
            raise TypeError(f"state {name} must be an exact numpy.ndarray")
        if array.dtype != np.dtype(np.float64):
            raise TypeError(f"state {name} must use float64")
        if not array.flags.c_contiguous or array.flags.writeable:
            raise ValueError(f"state {name} must be C-contiguous and immutable")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"state {name} must be finite")
    if positions.ndim != 2 or positions.shape[1:] != (3,):
        raise ValueError("state positions must have shape (n, 3)")
    if gamma.shape != positions.shape:
        raise ValueError("state gamma must match positions")
    if sigma.ndim != 1 or sigma.shape != (positions.shape[0],):
        raise ValueError("state sigma must have shape (n,)")
    if np.any(sigma <= 0.0):
        raise ValueError("state sigma must be strictly positive")
    return positions, gamma, sigma


@dataclass(frozen=True, slots=True)
class SheetPanel:
    """One causal quadrilateral release panel in GP1 coordinates."""

    upstream_left: tuple[float, float, float]
    upstream_right: tuple[float, float, float]
    downstream_left: tuple[float, float, float]
    downstream_right: tuple[float, float, float]
    circulation_m2_s: float
    release_index: int


@dataclass(frozen=True, slots=True)
class IncrementalLineage:
    """Birth provenance for one append-only contribution particle."""

    particle_id: tuple[Any, ...]
    release_index: int
    role: ParticleRole
    source_edge: EdgeKey
    parent_particle_id: tuple[Any, ...] | None
    source_lineage: ParticleLineage | None
    physical_owner: str = RING_PHYSICAL_OWNER
    owner_state: str = DIAGNOSTIC_SHADOW_OWNER


@dataclass(frozen=True, slots=True)
class ParticleSnapshot:
    """Read-only particle state used by direct and collapsed comparators."""

    positions: FloatArray
    gamma: FloatArray
    sigma: FloatArray
    particle_ids: tuple[tuple[Any, ...], ...]
    lineage: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class IncrementalSheetState:
    """Append-only contribution view and its current live-boundary facts."""

    interface_id: str
    positions: FloatArray
    gamma: FloatArray
    sigma: FloatArray
    particle_ids: tuple[tuple[Any, ...], ...]
    lineage: tuple[IncrementalLineage, ...]
    panels: tuple[SheetPanel, ...]
    downstream_particle_indices: tuple[int, ...]
    clone_pairs: tuple[tuple[int, int], ...]
    smoothing_radius_m: float
    target_spacing_m: float


@dataclass(frozen=True, slots=True)
class AppendDiagnostics:
    """Mechanically recomputed facts for one contribution append."""

    prefix_particle_count: int
    appended_particle_count: int
    cloned_particle_count: int
    excluded_fresh_upstream_count: int
    prefix_bitwise_unchanged: bool
    max_clone_position_abs: float
    max_clone_sigma_abs: float
    max_clone_gamma_relation_abs: float


@dataclass(frozen=True, slots=True)
class AppendResult:
    """Transactional result of a start or append operation."""

    state: IncrementalSheetState
    diagnostics: AppendDiagnostics


def make_panel(
    upstream_left: ArrayLike,
    upstream_right: ArrayLike,
    downstream_left: ArrayLike,
    downstream_right: ArrayLike,
    circulation_m2_s: float,
    *,
    release_index: int = 1,
) -> SheetPanel:
    """Build and validate one nondegenerate causal release panel."""

    panel = SheetPanel(
        upstream_left=_point("upstream_left", upstream_left),
        upstream_right=_point("upstream_right", upstream_right),
        downstream_left=_point("downstream_left", downstream_left),
        downstream_right=_point("downstream_right", downstream_right),
        circulation_m2_s=_finite_real("circulation_m2_s", circulation_m2_s),
        release_index=_positive_integer("release_index", release_index),
    )
    if panel.circulation_m2_s == 0.0:
        raise ValueError("circulation_m2_s must be nonzero")
    # Ring traversal is UL -> UR -> DR -> DL.  Keep the check in that exact
    # order; swapping DL/DR here would silently miss a collapsed side edge.
    points = tuple(
        np.asarray(value, dtype=np.float64)
        for value in (
            panel.upstream_left,
            panel.upstream_right,
            panel.downstream_right,
            panel.downstream_left,
        )
    )
    if any(
        float(np.linalg.norm(points[(index + 1) % 4] - points[index])) <= 0.0
        for index in range(4)
    ):
        raise ValueError("panel edges must have positive length")
    if len({tuple(point) for point in points}) != 4:
        raise ValueError("panel vertices must be distinct")
    upstream_span = points[1] - points[0]
    left_chord = points[3] - points[0]
    downstream_span = points[2] - points[3]
    right_chord = points[2] - points[1]
    upstream_normal = np.cross(upstream_span, left_chord)
    downstream_normal = np.cross(downstream_span, right_chord)
    if (
        float(np.linalg.norm(upstream_normal)) <= 0.0
        or float(np.linalg.norm(downstream_normal)) <= 0.0
        or float(np.dot(upstream_normal, downstream_normal)) <= 0.0
    ):
        raise ValueError("panel must have nonzero area at both span stations")
    return panel


def _validate_panel(panel: object, *, expected_release_index: int) -> SheetPanel:
    if type(panel) is not SheetPanel:
        raise TypeError("panel must be a SheetPanel")
    rebuilt = make_panel(
        panel.upstream_left,
        panel.upstream_right,
        panel.downstream_left,
        panel.downstream_right,
        panel.circulation_m2_s,
        release_index=panel.release_index,
    )
    if rebuilt != panel:
        raise ValueError("panel contains noncanonical values")
    if panel.release_index != expected_release_index:
        raise ValueError(
            f"panel release_index must be {expected_release_index}; "
            f"got {panel.release_index}"
        )
    return panel


def _panel_node_ids(panel: SheetPanel) -> dict[str, str]:
    prefix = f"v5h8:release:{panel.release_index}"
    return {
        "upstream_left": f"{prefix}:upstream:left",
        "upstream_right": f"{prefix}:upstream:right",
        "downstream_left": f"{prefix}:downstream:left",
        "downstream_right": f"{prefix}:downstream:right",
    }


def _panel_edge_keys(panel: SheetPanel) -> dict[str, EdgeKey]:
    ids = _panel_node_ids(panel)
    return {
        "upstream": canonical_edge_key(ids["upstream_left"], ids["upstream_right"]),
        "right_tip": canonical_edge_key(ids["upstream_right"], ids["downstream_right"]),
        "downstream": canonical_edge_key(
            ids["downstream_left"], ids["downstream_right"]
        ),
        "left_tip": canonical_edge_key(ids["upstream_left"], ids["downstream_left"]),
    }


def _deposit_panel(
    panel: SheetPanel,
    smoothing_radius_m: float,
    target_spacing_m: float,
) -> tuple[ParticleSnapshot, dict[str, EdgeKey]]:
    ids = _panel_node_ids(panel)
    nodes = (
        BridgeNode(ids["upstream_left"], panel.upstream_left),
        BridgeNode(ids["upstream_right"], panel.upstream_right),
        BridgeNode(ids["downstream_right"], panel.downstream_right),
        BridgeNode(ids["downstream_left"], panel.downstream_left),
    )
    ring = DirectedRing(
        ring_id=f"v5h8:ring:{panel.release_index}",
        node_ids=(
            ids["upstream_left"],
            ids["upstream_right"],
            ids["downstream_right"],
            ids["downstream_left"],
        ),
        circulation=panel.circulation_m2_s,
    )
    graph = assemble_ring_edge_graph(nodes, (ring,))
    result = deposit_edge_graph_prescribed_sigma_and_spacing(
        graph,
        smoothing_radius=smoothing_radius_m,
        target_spacing=target_spacing_m,
        step=panel.release_index,
    )
    snapshot = _snapshot(
        result.positions,
        result.gamma,
        result.sigma,
        tuple(result.particle_ids),
        tuple(result.lineage),
    )
    return snapshot, _panel_edge_keys(panel)


def _role_for_edge(edge: EdgeKey, keys: dict[str, EdgeKey]) -> ParticleRole:
    for name, role in (
        ("upstream", "fresh_upstream"),
        ("downstream", "fresh_downstream"),
        ("left_tip", "fresh_left_tip"),
        ("right_tip", "fresh_right_tip"),
    ):
        if edge == keys[name]:
            return role  # type: ignore[return-value]
    raise ValueError("deposited particle references an unknown panel edge")


def _snapshot(
    positions: ArrayLike,
    gamma: ArrayLike,
    sigma: ArrayLike,
    particle_ids: Sequence[tuple[Any, ...]],
    lineage: Sequence[object],
) -> ParticleSnapshot:
    position_array = _readonly_array("positions", positions, ndim=2)
    gamma_array = _readonly_array("gamma", gamma, ndim=2)
    sigma_array = _readonly_array("sigma", sigma, ndim=1)
    if position_array.shape[1:] != (3,):
        raise ValueError("positions must have shape (n, 3)")
    if gamma_array.shape != position_array.shape:
        raise ValueError("gamma must have the same shape as positions")
    if sigma_array.shape != (position_array.shape[0],):
        raise ValueError("sigma must have shape (n,)")
    if np.any(sigma_array <= 0.0):
        raise ValueError("sigma must be strictly positive")
    ids = tuple(particle_ids)
    records = tuple(lineage)
    if len(ids) != position_array.shape[0] or len(records) != len(ids):
        raise ValueError("particle arrays, IDs, and lineage must have equal length")
    if len(set(ids)) != len(ids):
        raise ValueError("particle IDs must be unique")
    return ParticleSnapshot(
        positions=position_array,
        gamma=gamma_array,
        sigma=sigma_array,
        particle_ids=ids,
        lineage=records,
    )


def _state_from_snapshot(
    snapshot: ParticleSnapshot,
    *,
    panels: tuple[SheetPanel, ...],
    lineage: tuple[IncrementalLineage, ...],
    downstream_particle_indices: tuple[int, ...],
    clone_pairs: tuple[tuple[int, int], ...],
    smoothing_radius_m: float,
    target_spacing_m: float,
) -> IncrementalSheetState:
    count = snapshot.positions.shape[0]
    if len(lineage) != count:
        raise ValueError("incremental lineage length disagrees with particle state")
    if len(set(downstream_particle_indices)) != len(downstream_particle_indices):
        raise ValueError("downstream particle indices must be unique")
    if any(index < 0 or index >= count for index in downstream_particle_indices):
        raise ValueError("downstream particle index is out of range")
    if any(old < 0 or clone <= old or clone >= count for old, clone in clone_pairs):
        raise ValueError("clone-pair indices are invalid")
    return IncrementalSheetState(
        interface_id=INTERFACE_ID,
        positions=snapshot.positions,
        gamma=snapshot.gamma,
        sigma=snapshot.sigma,
        particle_ids=snapshot.particle_ids,
        lineage=lineage,
        panels=panels,
        downstream_particle_indices=downstream_particle_indices,
        clone_pairs=clone_pairs,
        smoothing_radius_m=smoothing_radius_m,
        target_spacing_m=target_spacing_m,
    )


def start_incremental_sheet(
    panel: SheetPanel,
    smoothing_radius_m: float,
    target_spacing_m: float,
    *,
    particle_cap: int = 1_000,
) -> AppendResult:
    """Create the first append-only panel contribution."""

    panel = _validate_panel(panel, expected_release_index=1)
    sigma = _positive_real("smoothing_radius_m", smoothing_radius_m)
    spacing = _positive_real("target_spacing_m", target_spacing_m)
    cap = _positive_integer("particle_cap", particle_cap)
    deposited, keys = _deposit_panel(panel, sigma, spacing)
    count = deposited.positions.shape[0]
    if count > cap:
        raise RuntimeError("v5h8 particle cap exceeded")
    incremental_lineage = tuple(
        IncrementalLineage(
            particle_id=particle_id,
            release_index=panel.release_index,
            role=_role_for_edge(record.source_edge, keys),
            source_edge=record.source_edge,
            parent_particle_id=None,
            source_lineage=record,
        )
        for particle_id, record in zip(
            deposited.particle_ids, deposited.lineage, strict=True
        )
    )
    downstream = tuple(
        index
        for index, record in enumerate(incremental_lineage)
        if record.source_edge == keys["downstream"]
    )
    if not downstream:
        raise FloatingPointError("first panel has no downstream boundary particles")
    state = _state_from_snapshot(
        deposited,
        panels=(panel,),
        lineage=incremental_lineage,
        downstream_particle_indices=downstream,
        clone_pairs=(),
        smoothing_radius_m=sigma,
        target_spacing_m=spacing,
    )
    return AppendResult(
        state=state,
        diagnostics=AppendDiagnostics(
            prefix_particle_count=0,
            appended_particle_count=count,
            cloned_particle_count=0,
            excluded_fresh_upstream_count=0,
            prefix_bitwise_unchanged=True,
            max_clone_position_abs=0.0,
            max_clone_sigma_abs=0.0,
            max_clone_gamma_relation_abs=0.0,
        ),
    )


def _validate_state(state: object) -> IncrementalSheetState:
    if type(state) is not IncrementalSheetState:
        raise TypeError("state must be an IncrementalSheetState")
    if type(state.interface_id) is not str or state.interface_id != INTERFACE_ID:
        raise ValueError("state interface_id is invalid")
    positions, gamma, sigma = _validate_readonly_particle_arrays(
        state.positions,
        state.gamma,
        state.sigma,
    )
    if type(state.panels) is not tuple:
        raise TypeError("state panels must be a tuple")
    if not state.panels:
        raise ValueError("state must contain at least one panel")
    for release_index, panel in enumerate(state.panels, start=1):
        _validate_panel(panel, expected_release_index=release_index)
        if release_index > 1:
            previous = state.panels[release_index - 2]
            if not (
                np.array_equal(previous.downstream_left, panel.upstream_left)
                and np.array_equal(previous.downstream_right, panel.upstream_right)
            ):
                raise ValueError("state panels do not share an exact live boundary")
    _positive_real("state.smoothing_radius_m", state.smoothing_radius_m)
    _positive_real("state.target_spacing_m", state.target_spacing_m)

    if type(state.particle_ids) is not tuple or type(state.lineage) is not tuple:
        raise TypeError("state particle IDs and lineage must be tuples")
    if len(positions) != len(state.particle_ids) or len(gamma) != len(state.lineage):
        raise ValueError("state particle arrays, IDs, and lineage disagree")
    if any(type(particle_id) is not tuple for particle_id in state.particle_ids):
        raise TypeError("state particle IDs must be tuples")
    try:
        unique_id_count = len(set(state.particle_ids))
    except TypeError as error:
        raise TypeError("state particle IDs must be hashable") from error
    if unique_id_count != len(state.particle_ids):
        raise ValueError("state particle IDs must be unique")
    allowed_roles = {
        "fresh_upstream",
        "fresh_downstream",
        "fresh_left_tip",
        "fresh_right_tip",
        "inherited_upstream_counter",
    }
    for index, record in enumerate(state.lineage):
        if type(record) is not IncrementalLineage:
            raise TypeError("state lineage records must be IncrementalLineage")
        if record.particle_id != state.particle_ids[index]:
            raise ValueError("state lineage particle ID disagrees with particle state")
        if type(
            record.release_index
        ) is not int or not 1 <= record.release_index <= len(state.panels):
            raise ValueError("state lineage release index is invalid")
        if record.role not in allowed_roles:
            raise ValueError("state lineage role is invalid")
        if record.physical_owner != RING_PHYSICAL_OWNER:
            raise ValueError("ring must remain the physical owner")
        if record.owner_state != DIAGNOSTIC_SHADOW_OWNER:
            raise ValueError("particle state must remain diagnostic_shadow")
        if type(record.source_edge) is not tuple or len(record.source_edge) != 2:
            raise TypeError("state lineage source_edge is invalid")
        panel_keys = _panel_edge_keys(state.panels[record.release_index - 1])
        if record.role == "inherited_upstream_counter":
            if record.source_lineage is not None or record.parent_particle_id is None:
                raise ValueError("clone lineage parent/source contract is invalid")
            if record.source_edge != panel_keys["upstream"]:
                raise ValueError("clone lineage must use its release upstream edge")
            if (
                len(record.particle_id) != 4
                or record.particle_id[0] != CLONE_PARTICLE_SCHEMA
                or record.particle_id[1] != record.release_index
                or type(record.particle_id[2]) is not int
                or record.particle_id[3] != record.parent_particle_id
            ):
                raise ValueError("clone particle ID does not encode its provenance")
        else:
            if type(record.source_lineage) is not ParticleLineage:
                raise TypeError("fresh lineage must retain bridge birth provenance")
            if record.parent_particle_id is not None:
                raise ValueError("fresh lineage must not claim a clone parent")
            source = record.source_lineage
            if (
                source.particle_id != record.particle_id
                or source.source_edge != record.source_edge
                or source.step != record.release_index
                or source.physical_owner != RING_PHYSICAL_OWNER
                or source.owner_state != DIAGNOSTIC_SHADOW_OWNER
            ):
                raise ValueError("fresh bridge lineage provenance is inconsistent")
            if record.role != _role_for_edge(record.source_edge, panel_keys):
                raise ValueError("fresh lineage role disagrees with its panel edge")

    if type(state.downstream_particle_indices) is not tuple:
        raise TypeError("downstream particle indices must be a tuple")
    downstream = state.downstream_particle_indices
    if not downstream:
        raise ValueError("state must expose a live downstream boundary")
    if any(type(index) is not int for index in downstream):
        raise TypeError("downstream particle indices must be exact integers")
    if len(set(downstream)) != len(downstream):
        raise ValueError("downstream particle indices must be unique")
    if any(index < 0 or index >= len(positions) for index in downstream):
        raise ValueError("downstream particle index is out of range")
    expected_downstream = tuple(
        index
        for index, record in enumerate(state.lineage)
        if record.role == "fresh_downstream"
        and record.release_index == len(state.panels)
    )
    downstream_edges = {state.lineage[index].source_edge for index in downstream}
    if downstream != expected_downstream or len(downstream_edges) != 1:
        raise ValueError("live downstream indices do not identify the latest boundary")

    if type(state.clone_pairs) is not tuple:
        raise TypeError("clone_pairs must be a tuple")
    seen_clone_indices: set[int] = set()
    clone_indices: list[int] = []
    clone_parents_by_release: dict[int, list[int]] = {}
    for pair in state.clone_pairs:
        if (
            type(pair) is not tuple
            or len(pair) != 2
            or any(type(index) is not int for index in pair)
        ):
            raise TypeError("clone_pairs entries must be exact two-integer tuples")
        old_index, clone_index = pair
        if not 0 <= old_index < clone_index < len(positions):
            raise ValueError("clone-pair indices are out of order or range")
        if old_index in seen_clone_indices or clone_index in seen_clone_indices:
            raise ValueError("clone pairs must be disjoint")
        seen_clone_indices.update(pair)
        clone_indices.append(clone_index)
        clone_record = state.lineage[clone_index]
        if (
            clone_record.role != "inherited_upstream_counter"
            or clone_record.parent_particle_id != state.particle_ids[old_index]
        ):
            raise ValueError("clone lineage does not identify its live parent")
        if not np.array_equal(positions[clone_index], positions[old_index]):
            raise ValueError("clone position differs from its live parent")
        if sigma[clone_index] != sigma[old_index]:
            raise ValueError("clone sigma differs from its live parent")
        release_index = clone_record.release_index
        if release_index <= 1:
            raise ValueError("the first release cannot contain an inherited clone")
        current = state.panels[release_index - 1].circulation_m2_s
        previous = state.panels[release_index - 2].circulation_m2_s
        expected_gamma = -(current / previous) * gamma[old_index]
        if float(np.max(np.abs(gamma[clone_index] - expected_gamma))) > 1.0e-14:
            raise ValueError("clone gamma does not close its inherited-basis relation")
        clone_parents_by_release.setdefault(release_index, []).append(old_index)
    expected_clone_indices = tuple(
        index
        for index, record in enumerate(state.lineage)
        if record.role == "inherited_upstream_counter"
    )
    if tuple(clone_indices) != expected_clone_indices:
        raise ValueError(
            "clone_pairs do not cover every inherited counter exactly once"
        )
    for release_index in range(2, len(state.panels) + 1):
        expected_parents = [
            index
            for index, record in enumerate(state.lineage)
            if record.role == "fresh_downstream"
            and record.release_index == release_index - 1
        ]
        if clone_parents_by_release.get(release_index, []) != expected_parents:
            raise ValueError(
                "clone parent set does not equal the preceding live boundary"
            )
    return state


def append_live_basis_panel(
    state: IncrementalSheetState,
    downstream_left: ArrayLike,
    downstream_right: ArrayLike,
    circulation_m2_s: float,
    *,
    particle_cap: int = 1_000,
) -> AppendResult:
    """Append one panel by cloning, never redepositing, its upstream basis."""

    parent = _validate_state(state)
    cap = _positive_integer("particle_cap", particle_cap)
    if cap <= parent.positions.shape[0]:
        raise RuntimeError("v5h8 particle cap exceeded before append")
    circulation = _finite_real("circulation_m2_s", circulation_m2_s)
    previous_circulation = parent.panels[-1].circulation_m2_s
    if (
        circulation == 0.0
        or np.signbit(circulation) != np.signbit(previous_circulation)
        or abs(circulation) <= abs(previous_circulation)
    ):
        raise ValueError("circulation must retain its sign and increase in magnitude")
    release_index = len(parent.panels) + 1
    panel = make_panel(
        parent.panels[-1].downstream_left,
        parent.panels[-1].downstream_right,
        downstream_left,
        downstream_right,
        circulation,
        release_index=release_index,
    )
    deposited, keys = _deposit_panel(
        panel,
        parent.smoothing_radius_m,
        parent.target_spacing_m,
    )
    deposited_lineage = tuple(deposited.lineage)
    upstream_indices = tuple(
        index
        for index, record in enumerate(deposited_lineage)
        if record.source_edge == keys["upstream"]
    )
    if not upstream_indices:
        raise FloatingPointError("new panel has no upstream particles to exclude")
    old_boundary = parent.downstream_particle_indices
    if not old_boundary:
        raise FloatingPointError("parent has no live downstream boundary")

    scale = -circulation / previous_circulation
    clone_positions = np.ascontiguousarray(parent.positions[list(old_boundary)])
    clone_sigma = np.ascontiguousarray(parent.sigma[list(old_boundary)])
    clone_gamma = np.ascontiguousarray(parent.gamma[list(old_boundary)] * scale)
    clone_ids = tuple(
        (
            CLONE_PARTICLE_SCHEMA,
            release_index,
            clone_order,
            parent.particle_ids[old_index],
        )
        for clone_order, old_index in enumerate(old_boundary)
    )
    clone_lineage = tuple(
        IncrementalLineage(
            particle_id=particle_id,
            release_index=release_index,
            role="inherited_upstream_counter",
            source_edge=keys["upstream"],
            parent_particle_id=parent.particle_ids[old_index],
            source_lineage=None,
        )
        for particle_id, old_index in zip(clone_ids, old_boundary, strict=True)
    )

    upstream_set = set(upstream_indices)
    kept_indices = tuple(
        index
        for index in range(deposited.positions.shape[0])
        if index not in upstream_set
    )
    fresh_positions = np.ascontiguousarray(deposited.positions[list(kept_indices)])
    fresh_gamma = np.ascontiguousarray(deposited.gamma[list(kept_indices)])
    fresh_sigma = np.ascontiguousarray(deposited.sigma[list(kept_indices)])
    fresh_ids = tuple(deposited.particle_ids[index] for index in kept_indices)
    fresh_lineage = tuple(
        IncrementalLineage(
            particle_id=deposited.particle_ids[index],
            release_index=release_index,
            role=_role_for_edge(deposited_lineage[index].source_edge, keys),
            source_edge=deposited_lineage[index].source_edge,
            parent_particle_id=None,
            source_lineage=deposited_lineage[index],
        )
        for index in kept_indices
    )

    prefix_count = parent.positions.shape[0]
    total_count = prefix_count + len(clone_ids) + len(fresh_ids)
    if total_count > cap:
        raise RuntimeError("v5h8 particle cap exceeded during append")
    all_ids = parent.particle_ids + clone_ids + fresh_ids
    if len(set(all_ids)) != len(all_ids):
        raise FloatingPointError("append produced duplicate particle IDs")
    positions = np.vstack((parent.positions, clone_positions, fresh_positions))
    gamma = np.vstack((parent.gamma, clone_gamma, fresh_gamma))
    sigma = np.concatenate((parent.sigma, clone_sigma, fresh_sigma))
    snapshot = _snapshot(
        positions,
        gamma,
        sigma,
        all_ids,
        parent.lineage + clone_lineage + fresh_lineage,
    )

    clone_pairs = parent.clone_pairs + tuple(
        (old_index, prefix_count + clone_order)
        for clone_order, old_index in enumerate(old_boundary)
    )
    deposited_to_new: dict[int, int] = {
        deposited_index: prefix_count + len(clone_ids) + fresh_order
        for fresh_order, deposited_index in enumerate(kept_indices)
    }
    downstream = tuple(
        deposited_to_new[index]
        for index, record in enumerate(deposited_lineage)
        if record.source_edge == keys["downstream"]
    )
    if not downstream:
        raise FloatingPointError("new panel has no downstream boundary particles")
    state_after = _state_from_snapshot(
        snapshot,
        panels=parent.panels + (panel,),
        lineage=parent.lineage + clone_lineage + fresh_lineage,
        downstream_particle_indices=downstream,
        clone_pairs=clone_pairs,
        smoothing_radius_m=parent.smoothing_radius_m,
        target_spacing_m=parent.target_spacing_m,
    )

    prefix_exact = (
        state_after.positions[:prefix_count].tobytes() == parent.positions.tobytes()
        and state_after.gamma[:prefix_count].tobytes() == parent.gamma.tobytes()
        and state_after.sigma[:prefix_count].tobytes() == parent.sigma.tobytes()
        and state_after.particle_ids[:prefix_count] == parent.particle_ids
        and state_after.lineage[:prefix_count] == parent.lineage
    )
    cloned_indices = np.arange(prefix_count, prefix_count + len(clone_ids))
    old_indices = np.asarray(old_boundary, dtype=np.int64)
    max_position = float(
        np.max(
            np.abs(
                state_after.positions[cloned_indices]
                - state_after.positions[old_indices]
            )
        )
    )
    max_sigma = float(
        np.max(
            np.abs(state_after.sigma[cloned_indices] - state_after.sigma[old_indices])
        )
    )
    max_gamma = float(
        np.max(
            np.abs(
                state_after.gamma[cloned_indices]
                - scale * state_after.gamma[old_indices]
            )
        )
    )
    return AppendResult(
        state=state_after,
        diagnostics=AppendDiagnostics(
            prefix_particle_count=prefix_count,
            appended_particle_count=total_count - prefix_count,
            cloned_particle_count=len(clone_ids),
            excluded_fresh_upstream_count=len(upstream_indices),
            prefix_bitwise_unchanged=prefix_exact,
            max_clone_position_abs=max_position,
            max_clone_sigma_abs=max_sigma,
            max_clone_gamma_relation_abs=max_gamma,
        ),
    )


def affine_transport_state(
    state: IncrementalSheetState,
    matrix: ArrayLike,
    translation: ArrayLike,
    sigma_scale: float,
) -> IncrementalSheetState:
    """Apply one deterministic affine material map to every live contribution."""

    parent = _validate_state(state)
    transform = _matrix("matrix", matrix)
    offset = np.asarray(_point("translation", translation), dtype=np.float64)
    scale = _positive_real("sigma_scale", sigma_scale)
    positions = np.ascontiguousarray(parent.positions @ transform.T + offset)
    gamma = np.ascontiguousarray(parent.gamma @ transform.T)
    sigma = np.ascontiguousarray(parent.sigma * scale)
    if not (
        np.all(np.isfinite(positions))
        and np.all(np.isfinite(gamma))
        and np.all(np.isfinite(sigma))
        and np.all(sigma > 0.0)
    ):
        raise FloatingPointError("affine transport produced an invalid particle state")

    def transform_point(value: tuple[float, float, float]) -> FloatArray:
        return transform @ np.asarray(value, dtype=np.float64) + offset

    panels = tuple(
        make_panel(
            transform_point(panel.upstream_left),
            transform_point(panel.upstream_right),
            transform_point(panel.downstream_left),
            transform_point(panel.downstream_right),
            panel.circulation_m2_s,
            release_index=panel.release_index,
        )
        for panel in parent.panels
    )
    snapshot = _snapshot(
        positions,
        gamma,
        sigma,
        parent.particle_ids,
        parent.lineage,
    )
    return _state_from_snapshot(
        snapshot,
        panels=panels,
        lineage=parent.lineage,
        downstream_particle_indices=parent.downstream_particle_indices,
        clone_pairs=parent.clone_pairs,
        smoothing_radius_m=parent.smoothing_radius_m,
        target_spacing_m=parent.target_spacing_m,
    )


def _connected_graph(panels: Sequence[SheetPanel]):
    panel_tuple = tuple(panels)
    if not panel_tuple:
        raise ValueError("panels must not be empty")
    if any(type(panel) is not SheetPanel for panel in panel_tuple):
        raise TypeError("panels must contain only SheetPanel objects")
    for index, panel in enumerate(panel_tuple, start=1):
        _validate_panel(panel, expected_release_index=index)
        if index > 1:
            previous = panel_tuple[index - 2]
            if not (
                np.array_equal(previous.downstream_left, panel.upstream_left)
                and np.array_equal(previous.downstream_right, panel.upstream_right)
            ):
                raise ValueError(
                    "adjacent panels must share exact boundary coordinates"
                )

    nodes: list[BridgeNode] = [
        BridgeNode("v5h8:plane:0:left", panel_tuple[0].upstream_left),
        BridgeNode("v5h8:plane:0:right", panel_tuple[0].upstream_right),
    ]
    for plane, panel in enumerate(panel_tuple, start=1):
        nodes.extend(
            (
                BridgeNode(f"v5h8:plane:{plane}:left", panel.downstream_left),
                BridgeNode(f"v5h8:plane:{plane}:right", panel.downstream_right),
            )
        )
    rings = tuple(
        DirectedRing(
            ring_id=f"v5h8:connected:ring:{index}",
            node_ids=(
                f"v5h8:plane:{index - 1}:left",
                f"v5h8:plane:{index - 1}:right",
                f"v5h8:plane:{index}:right",
                f"v5h8:plane:{index}:left",
            ),
            circulation=panel.circulation_m2_s,
        )
        for index, panel in enumerate(panel_tuple, start=1)
    )
    return assemble_ring_edge_graph(tuple(nodes), rings)


def direct_connected_redeposit(
    panels: Sequence[SheetPanel],
    smoothing_radius_m: float,
    target_spacing_m: float,
) -> ParticleSnapshot:
    """Materialize the canonical connected graph using fresh birth particles."""

    sigma = _positive_real("smoothing_radius_m", smoothing_radius_m)
    spacing = _positive_real("target_spacing_m", target_spacing_m)
    graph = _connected_graph(panels)
    result = deposit_edge_graph_prescribed_sigma_and_spacing(
        graph,
        smoothing_radius=sigma,
        target_spacing=spacing,
        step=len(tuple(panels)),
    )
    return _snapshot(
        result.positions,
        result.gamma,
        result.sigma,
        result.particle_ids,
        result.lineage,
    )


def fresh_geometry_redeposit(state: IncrementalSheetState) -> ParticleSnapshot:
    """Negative control: discard live material support and redeposit geometry."""

    parent = _validate_state(state)
    return direct_connected_redeposit(
        parent.panels,
        parent.smoothing_radius_m,
        parent.target_spacing_m,
    )


def collapse_live_basis_pairs(state: IncrementalSheetState) -> ParticleSnapshot:
    """Algebraically collapse co-located counter-pairs into a derived net view."""

    parent = _validate_state(state)
    gamma = np.array(parent.gamma, dtype=np.float64, order="C", copy=True)
    removed: set[int] = set()
    for old_index, clone_index in parent.clone_pairs:
        if clone_index in removed or old_index in removed:
            raise ValueError("clone pairs must be disjoint")
        if not np.array_equal(
            parent.positions[old_index], parent.positions[clone_index]
        ):
            raise ValueError("clone pair positions are not bitwise identical")
        if parent.sigma[old_index] != parent.sigma[clone_index]:
            raise ValueError("clone pair sigma values are not bitwise identical")
        gamma[old_index] += gamma[clone_index]
        removed.add(clone_index)
    kept = tuple(index for index in range(len(gamma)) if index not in removed)
    ids = tuple(parent.particle_ids[index] for index in kept)
    lineage = tuple(parent.lineage[index] for index in kept)
    return _snapshot(
        parent.positions[list(kept)],
        gamma[list(kept)],
        parent.sigma[list(kept)],
        ids,
        lineage,
    )


def _particle_arrays(value: object) -> tuple[FloatArray, FloatArray, FloatArray]:
    if type(value) is IncrementalSheetState:
        state = _validate_state(value)
        return state.positions, state.gamma, state.sigma
    if type(value) is not ParticleSnapshot:
        raise TypeError("value must be a ParticleSnapshot or IncrementalSheetState")
    positions, gamma, sigma = _validate_readonly_particle_arrays(
        value.positions,
        value.gamma,
        value.sigma,
    )
    if type(value.particle_ids) is not tuple or type(value.lineage) is not tuple:
        raise TypeError("snapshot IDs and lineage must be tuples")
    if len(value.particle_ids) != len(positions) or len(value.lineage) != len(
        positions
    ):
        raise ValueError("snapshot arrays, IDs, and lineage disagree")
    if any(type(particle_id) is not tuple for particle_id in value.particle_ids):
        raise TypeError("snapshot particle IDs must be tuples")
    try:
        unique_id_count = len(set(value.particle_ids))
    except TypeError as error:
        raise TypeError("snapshot particle IDs must be hashable") from error
    if unique_id_count != len(value.particle_ids):
        raise ValueError("snapshot particle IDs must be unique")
    return positions, gamma, sigma


def particle_velocity(value: object, probes: ArrayLike) -> FloatArray:
    """Evaluate the frozen Gaussian-erf particle field at finite GP1 probes."""

    positions, gamma, sigma = _particle_arrays(value)
    original = np.asarray(probes)
    if original.dtype.kind not in "iuf" or original.dtype.kind == "b":
        raise TypeError("probes must use a real numeric dtype")
    targets = np.asarray(original, dtype=np.float64)
    if targets.ndim != 2 or targets.shape[1:] != (3,):
        raise ValueError("probes must have shape (m, 3)")
    if not np.all(np.isfinite(targets)):
        raise ValueError("probes must contain only finite values")
    return direct_gaussian_erf_velocity_jacobian(
        positions,
        gamma,
        sigma,
        target_positions=targets,
    ).velocity


def particle_impulse(value: object) -> FloatArray:
    """Return ``0.5 * sum(x cross gamma)`` using stable component sums."""

    positions, gamma, _ = _particle_arrays(value)
    cross = np.cross(positions, gamma)
    impulse = np.asarray(
        [0.5 * fsum(float(item) for item in cross[:, axis]) for axis in range(3)],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(impulse)):
        raise FloatingPointError("particle impulse is non-finite")
    return impulse


__all__ = [
    "AppendDiagnostics",
    "AppendResult",
    "IncrementalLineage",
    "IncrementalSheetState",
    "ParticleSnapshot",
    "SheetPanel",
    "affine_transport_state",
    "append_live_basis_panel",
    "collapse_live_basis_pairs",
    "direct_connected_redeposit",
    "fresh_geometry_redeposit",
    "make_panel",
    "particle_impulse",
    "particle_velocity",
    "start_incremental_sheet",
]
