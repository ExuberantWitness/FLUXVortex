"""Transactional Ptera adapter for Hirato's material LEV sheet state.

This module owns geometry and history only.  It deliberately contains no AIC,
pseudovortex, trailing-edge, load, or vortex-core policy.  The underlying
``HiratoSheetShadow`` remains the single implementation of the Eq. 7 remesh
and Eq. 24 material-point history.

``HiratoSheetShadow`` stores a readable forward traversal
``[front-left, front-right, aft-right, aft-left]``.  Ptera's positive ring
circulation traverses that readable order in reverse.  Consequently material
rings are exposed to Ptera after the exact self-inverse permutation
``[1, 0, 3, 2]``; circulation is never sign-flipped.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from claim_runtime.hirato_equations import HiratoEquationError
from claim_runtime.hirato_shadow import HiratoSheetShadow


PTERA_REVERSE_RING_TRAVERSAL: Final = (1, 0, 3, 2)


def _finite_array(value: Any, *, name: str, ndim: int) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != ndim:
        raise HiratoEquationError(f"{name} must be {ndim}D, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise HiratoEquationError(f"{name} must contain only finite values")
    return array


def _boolean_vector(value: Any, *, name: str, size: int) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind != "b" or array.shape != (size,):
        raise HiratoEquationError(
            f"{name} must be a boolean array with shape {(size,)}, got "
            f"{array.shape} and dtype {array.dtype}"
        )
    return array.astype(bool, copy=True)


def _step(value: Any, *, name: str = "step") -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise HiratoEquationError(f"{name} must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise HiratoEquationError(f"{name} must be a non-negative integer")
    return result


def _positive_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise HiratoEquationError(f"{name} must be a positive finite scalar")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise HiratoEquationError(f"{name} must be a positive finite scalar")
    return result


def _bitwise_equal(left: np.ndarray, right: np.ndarray) -> bool:
    return (
        left.dtype == right.dtype
        and left.shape == right.shape
        and left.tobytes() == right.tobytes()
    )


def _ring_array(value: Any, *, name: str) -> np.ndarray:
    rings = _finite_array(value, name=name, ndim=3)
    if rings.shape[1:] != (4, 3):
        raise HiratoEquationError(
            f"{name} must have shape (n, 4, 3), got {rings.shape}"
        )
    return rings


def shadow_forward_to_ptera_rings(rings: Any) -> np.ndarray:
    """Reverse readable shadow rings for Ptera without changing ``Gamma``.

    The permutation is a traversal reversal, not a geometry reflection, and
    is self-inverse.  The returned array never aliases the caller's storage.
    """

    shadow_rings = _ring_array(rings, name="shadow_forward_rings")
    return shadow_rings[:, PTERA_REVERSE_RING_TRAVERSAL].copy()


def ptera_to_shadow_forward_rings(rings: Any) -> np.ndarray:
    """Convert Ptera-readable material rings back to shadow traversal."""

    ptera_rings = _ring_array(rings, name="ptera_rings")
    return ptera_rings[:, PTERA_REVERSE_RING_TRAVERSAL].copy()


def _ptera_to_shadow_vertex_values(values: Any, *, name: str) -> np.ndarray:
    array = _ring_array(values, name=name)
    return array[:, PTERA_REVERSE_RING_TRAVERSAL].copy()


def _shadow_to_ptera_vertex_values(values: Any, *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 3 or array.shape[1:] != (4, 3):
        raise HiratoEquationError(
            f"{name} must have shape (n, 4, 3), got {array.shape}"
        )
    return array[:, PTERA_REVERSE_RING_TRAVERSAL].copy()


@dataclass(frozen=True)
class PteraMaterialLEVView:
    """Ptera-ordered material-ring field at an explicit reference step."""

    rings_gp1_m: np.ndarray
    gamma_m2_s: np.ndarray
    strip: np.ndarray
    birth_step: np.ndarray
    age_steps: np.ndarray
    ring_id: np.ndarray
    sheet_id: np.ndarray
    reference_step: int


@dataclass(frozen=True)
class PteraMaterialLEVSnapshot:
    """Complete copy of adapter state, including transaction sequencing."""

    shadow_rings_forward_gp1_m: np.ndarray
    ptera_rings_gp1_m: np.ndarray
    gamma_m2_s: np.ndarray
    strip: np.ndarray
    birth_step: np.ndarray
    ring_id: np.ndarray
    sheet_id: np.ndarray
    previous_vertex_velocity_ptera_m_s: np.ndarray
    current_sheet_by_strip: np.ndarray
    active_last_step: np.ndarray
    next_ring_id: int
    next_sheet_id: int
    last_committed_step: int
    last_convected_step: int
    version: int


@dataclass(frozen=True)
class PteraEq24ConvectionReport:
    """Eq. 24 record converted into the same Ptera-readable traversal."""

    step: int
    bootstrap_vertex: np.ndarray
    displacement_gp1_m: np.ndarray


class PteraMaterialLEVProposal:
    """Uncommitted shedding transaction built entirely on a cloned sheet."""

    def __init__(
        self,
        *,
        owner: PteraNativeMaterialLEVState,
        candidate_shadow: HiratoSheetShadow,
        step: int,
        base_version: int,
        active: np.ndarray,
        created_ids: np.ndarray,
        active_strip_indices: np.ndarray,
        old_ring_count: int,
    ) -> None:
        self._owner = owner
        self._candidate_shadow = candidate_shadow
        self.step = int(step)
        self.base_version = int(base_version)
        self._active = active.copy()
        self._created_ids = created_ids.copy()
        self._active_strip_indices = active_strip_indices.copy()
        self._old_ring_count = int(old_ring_count)
        self._finalized = False

    @property
    def newborn_ptera_rings_gp1_m(self) -> np.ndarray:
        """New rings in ``created_ids`` order, safe for a Ptera solve."""

        positions = self._created_positions()
        return shadow_forward_to_ptera_rings(self._candidate_shadow.rings[positions])

    @property
    def active(self) -> np.ndarray:
        return self._active.copy()

    @property
    def created_ids(self) -> np.ndarray:
        return self._created_ids.copy()

    @property
    def active_strip_indices(self) -> np.ndarray:
        return self._active_strip_indices.copy()

    @property
    def is_finalized(self) -> bool:
        return self._finalized

    def _created_positions(self) -> np.ndarray:
        positions: list[int] = []
        for ring_id in self._created_ids:
            matches = np.flatnonzero(self._candidate_shadow.ring_id == ring_id)
            if matches.size != 1:
                raise RuntimeError("proposal contains an invalid created ring identity")
            positions.append(int(matches[0]))
        return np.asarray(positions, dtype=np.int64)

    def candidate_material_view(
        self,
        *,
        created_strengths_m2_s: Any | None = None,
    ) -> PteraMaterialLEVView:
        """Expose the uncommitted Eq. 7 candidate without publishing it.

        A coupled LESP solve needs the remeshed old sheet in its fixed
        right-hand side while the newborn rings remain unknown influence
        columns.  Supplying solved newborn strengths changes only the returned
        copy; neither this proposal nor its owner is modified until
        :meth:`commit` is called.
        """

        if self._finalized:
            raise RuntimeError("proposal has already been finalized")
        if self.base_version != self._owner._version:
            raise RuntimeError("proposal is stale because material state changed")
        gamma = self._candidate_shadow.gamma.copy()
        if created_strengths_m2_s is not None:
            strengths = _finite_array(
                created_strengths_m2_s,
                name="created_strengths_m2_s",
                ndim=1,
            )
            if strengths.shape != self._created_ids.shape:
                raise HiratoEquationError(
                    "created_strengths_m2_s must have one value per created ring"
                )
            gamma[self._created_positions()] = strengths
        ages = self.step - self._candidate_shadow.birth_step
        if np.any(ages < 0):
            raise RuntimeError("candidate contains a ring born in the future")
        return PteraMaterialLEVView(
            rings_gp1_m=shadow_forward_to_ptera_rings(self._candidate_shadow.rings),
            gamma_m2_s=gamma,
            strip=self._candidate_shadow.strip.copy(),
            birth_step=self._candidate_shadow.birth_step.copy(),
            age_steps=np.asarray(ages, dtype=np.int64),
            ring_id=self._candidate_shadow.ring_id.copy(),
            sheet_id=self._candidate_shadow.sheet_id.copy(),
            reference_step=self.step,
        )

    def commit(self, created_strengths_m2_s: Any) -> PteraMaterialLEVSnapshot:
        """Assign solved newborn strengths and atomically publish the clone."""

        if self._finalized:
            raise RuntimeError("proposal has already been finalized")
        strengths = _finite_array(
            created_strengths_m2_s,
            name="created_strengths_m2_s",
            ndim=1,
        )
        if strengths.shape != self._created_ids.shape:
            raise HiratoEquationError(
                "created_strengths_m2_s must have one value per created ring"
            )

        old_gamma = self._candidate_shadow.gamma[: self._old_ring_count].copy()
        self._candidate_shadow.assign_created_strengths(self._created_ids, strengths)
        if not _bitwise_equal(
            self._candidate_shadow.gamma[: self._old_ring_count], old_gamma
        ):
            raise RuntimeError("committing newborn strengths changed old circulation")
        positions = self._created_positions()
        if not _bitwise_equal(self._candidate_shadow.gamma[positions], strengths):
            raise RuntimeError("newborn strength assignment did not close by ring id")

        snapshot = self._owner._commit_proposal(self)
        self._finalized = True
        return snapshot

    def commit_and_convect(
        self,
        created_strengths_m2_s: Any,
        vertex_velocity_ptera_m_s: Any,
        *,
        dt_s: float,
    ) -> tuple[PteraMaterialLEVSnapshot, PteraEq24ConvectionReport]:
        """Atomically assign, convect, and publish one complete material step.

        The coupled solver must keep the Eq. 7 candidate private while Ptera
        finishes its current load and TE-wake operations.  Both the strength
        assignment and Eq. 24 update are therefore performed on another clone;
        the owner is changed only after every validation succeeds.
        """

        if self._finalized:
            raise RuntimeError("proposal has already been finalized")
        if self.base_version != self._owner._version:
            raise RuntimeError("proposal is stale because material state changed")
        strengths = _finite_array(
            created_strengths_m2_s,
            name="created_strengths_m2_s",
            ndim=1,
        )
        if strengths.shape != self._created_ids.shape:
            raise HiratoEquationError(
                "created_strengths_m2_s must have one value per created ring"
            )
        dt = _positive_float(dt_s, name="dt_s")
        candidate = copy.deepcopy(self._candidate_shadow)
        old_gamma = candidate.gamma[: self._old_ring_count].copy()
        candidate.assign_created_strengths(self._created_ids, strengths)
        if not _bitwise_equal(candidate.gamma[: self._old_ring_count], old_gamma):
            raise RuntimeError("assigning newborn strengths changed old circulation")

        velocity_ptera = _ring_array(
            vertex_velocity_ptera_m_s,
            name="vertex_velocity_ptera_m_s",
        )
        if velocity_ptera.shape != candidate.rings.shape:
            raise HiratoEquationError(
                "vertex_velocity_ptera_m_s must match the candidate material rings"
            )
        identity_before = (
            candidate.strip.copy(),
            candidate.birth_step.copy(),
            candidate.ring_id.copy(),
            candidate.sheet_id.copy(),
        )
        gamma_before_convection = candidate.gamma.copy()
        report = candidate.convect_eq24(
            _ptera_to_shadow_vertex_values(
                velocity_ptera,
                name="vertex_velocity_ptera_m_s",
            ),
            dt,
            step=self.step,
        )
        if not _bitwise_equal(candidate.gamma, gamma_before_convection):
            raise RuntimeError("Eq. 24 convection changed material circulation")
        for current, before in zip(
            (
                candidate.strip,
                candidate.birth_step,
                candidate.ring_id,
                candidate.sheet_id,
            ),
            identity_before,
            strict=True,
        ):
            if not np.array_equal(current, before):
                raise RuntimeError("Eq. 24 convection changed material identity")

        snapshot = self._owner._commit_convected_proposal(self, candidate)
        self._finalized = True
        return snapshot, PteraEq24ConvectionReport(
            step=report.step,
            bootstrap_vertex=report.bootstrap_vertex[
                :, PTERA_REVERSE_RING_TRAVERSAL
            ].copy(),
            displacement_gp1_m=_shadow_to_ptera_vertex_values(
                report.displacement,
                name="Eq24 displacement",
            ),
        )

    def commit_by_strip(self, gamma_lev_by_strip_m2_s: Any) -> PteraMaterialLEVSnapshot:
        """Commit a full strip vector while refusing nonzero inactive entries."""

        values = _finite_array(
            gamma_lev_by_strip_m2_s,
            name="gamma_lev_by_strip_m2_s",
            ndim=1,
        )
        if values.shape != (self._owner.ns,):
            raise HiratoEquationError(
                f"gamma_lev_by_strip_m2_s must have shape {(self._owner.ns,)}"
            )
        if np.any(values[~self._active] != 0.0):
            raise HiratoEquationError(
                "inactive strips must have exactly zero proposed circulation"
            )
        return self.commit(values[self._active_strip_indices])

    def abort(self) -> None:
        """Discard the clone; the owner is not touched."""

        if self._finalized:
            raise RuntimeError("proposal has already been finalized")
        self._finalized = True


class PteraNativeMaterialLEVState:
    """Fail-closed material LEV state with transactional Eq. 7 shedding."""

    def __init__(self, ns: int):
        if isinstance(ns, (bool, np.bool_)) or not isinstance(ns, (int, np.integer)):
            raise HiratoEquationError("ns must be a positive integer")
        if int(ns) <= 0:
            raise HiratoEquationError("ns must be a positive integer")
        self._shadow = HiratoSheetShadow(int(ns))
        self.ns = self._shadow.ns
        self._last_committed_step = -1
        self._last_convected_step = -1
        self._active_last_step = np.zeros(self.ns, dtype=bool)
        self._version = 0

    @property
    def last_committed_step(self) -> int:
        return self._last_committed_step

    @property
    def last_convected_step(self) -> int:
        return self._last_convected_step

    def snapshot(self) -> PteraMaterialLEVSnapshot:
        """Return a non-aliasing, complete state snapshot."""

        shadow = self._shadow.snapshot()
        return PteraMaterialLEVSnapshot(
            shadow_rings_forward_gp1_m=shadow.rings,
            ptera_rings_gp1_m=shadow_forward_to_ptera_rings(shadow.rings),
            gamma_m2_s=shadow.gamma,
            strip=shadow.strip,
            birth_step=shadow.birth_step,
            ring_id=shadow.ring_id,
            sheet_id=shadow.sheet_id,
            previous_vertex_velocity_ptera_m_s=_shadow_to_ptera_vertex_values(
                shadow.previous_vertex_velocity,
                name="previous_vertex_velocity",
            ),
            current_sheet_by_strip=self._shadow._current_sheet.copy(),
            active_last_step=self._active_last_step.copy(),
            next_ring_id=int(self._shadow._next_id),
            next_sheet_id=int(self._shadow._next_sheet_id),
            last_committed_step=self._last_committed_step,
            last_convected_step=self._last_convected_step,
            version=self._version,
        )

    def material_view(
        self, *, reference_step: int | None = None
    ) -> PteraMaterialLEVView:
        """Expose Ptera rings, circulation, identity, and integer material age."""

        if reference_step is None:
            reference = self._last_committed_step
        else:
            reference = _step(reference_step, name="reference_step")
        if self._shadow.rings.shape[0] and reference < int(
            np.max(self._shadow.birth_step)
        ):
            raise HiratoEquationError("reference_step precedes a material-ring birth")
        ages = (
            reference - self._shadow.birth_step
            if self._shadow.rings.shape[0]
            else np.empty(0, dtype=np.int64)
        )
        return PteraMaterialLEVView(
            rings_gp1_m=shadow_forward_to_ptera_rings(self._shadow.rings),
            gamma_m2_s=self._shadow.gamma.copy(),
            strip=self._shadow.strip.copy(),
            birth_step=self._shadow.birth_step.copy(),
            age_steps=np.asarray(ages, dtype=np.int64),
            ring_id=self._shadow.ring_id.copy(),
            sheet_id=self._shadow.sheet_id.copy(),
            reference_step=reference,
        )

    def indices_for_strip(self, strip: int) -> np.ndarray:
        """Return stable material-array positions owned by one strip."""

        if isinstance(strip, (bool, np.bool_)) or not isinstance(
            strip, (int, np.integer)
        ):
            raise HiratoEquationError("strip must be an integer")
        strip_index = int(strip)
        if strip_index < 0 or strip_index >= self.ns:
            raise HiratoEquationError(
                f"strip must lie in [0, {self.ns}), got {strip_index}"
            )
        return np.flatnonzero(self._shadow.strip == strip_index)

    def ring_ids_for_strip(self, strip: int) -> np.ndarray:
        """Return persistent ring identities for one strip."""

        return self._shadow.ring_id[self.indices_for_strip(strip)].copy()

    def propose_shed(
        self,
        *,
        step: int,
        leading_edges_gp1_m: Any,
        active: Any,
        new_sheet: Any | None = None,
        first_aft_edges_gp1_m: Any | None = None,
    ) -> PteraMaterialLEVProposal:
        """Build a zero-strength Eq. 7 proposal without mutating this state."""

        step_value = _step(step)
        expected_step = self._last_committed_step + 1
        if step_value != expected_step:
            raise HiratoEquationError(
                f"step sequence must advance by one: expected {expected_step}, "
                f"got {step_value}"
            )
        if (
            self._shadow.rings.shape[0]
            and self._last_convected_step < self._last_committed_step
        ):
            raise HiratoEquationError(
                "the previous committed material state must be convected before "
                "the next shedding proposal"
            )

        leading_edges = _finite_array(
            leading_edges_gp1_m, name="leading_edges_gp1_m", ndim=3
        )
        expected_edges = (self.ns, 2, 3)
        if leading_edges.shape != expected_edges:
            raise HiratoEquationError(
                f"leading_edges_gp1_m must have shape {expected_edges}, got "
                f"{leading_edges.shape}"
            )
        active_mask = _boolean_vector(active, name="active", size=self.ns)
        starts = (
            np.zeros(self.ns, dtype=bool)
            if new_sheet is None
            else _boolean_vector(new_sheet, name="new_sheet", size=self.ns)
        )
        if np.any(starts & ~active_mask):
            raise HiratoEquationError("new_sheet may only be true on active strips")
        first_edges = None
        if first_aft_edges_gp1_m is not None:
            first_edges = _finite_array(
                first_aft_edges_gp1_m,
                name="first_aft_edges_gp1_m",
                ndim=3,
            )
            if first_edges.shape != expected_edges:
                raise HiratoEquationError(
                    f"first_aft_edges_gp1_m must have shape {expected_edges}, "
                    f"got {first_edges.shape}"
                )

        before = self._shadow.snapshot()
        old_ring_count = before.rings.shape[0]
        allowed_geometry_change = np.zeros(before.rings.shape, dtype=bool)
        for strip_index in np.flatnonzero(active_mask & ~starts):
            newest = self._shadow._newest_index(int(strip_index))
            if newest is not None:
                allowed_geometry_change[newest, 0:2, :] = True

        candidate = copy.deepcopy(self._shadow)
        report = candidate.shed(
            step=step_value,
            leading_edges=leading_edges,
            gamma_now=np.zeros(self.ns, dtype=float),
            active=active_mask,
            first_aft_edges=first_edges,
            new_sheet=starts,
        )
        active_indices = np.flatnonzero(active_mask)
        if report.created_ids.shape != active_indices.shape:
            raise RuntimeError(
                "shadow shedding did not create one ring per active strip"
            )
        if not np.array_equal(candidate.strip[old_ring_count:], active_indices):
            raise RuntimeError("shadow shedding changed active-strip ordering")
        if not _bitwise_equal(candidate.gamma[:old_ring_count], before.gamma):
            raise RuntimeError("Eq. 7 remeshing changed old circulation")
        if not np.array_equal(
            candidate.gamma[old_ring_count:], np.zeros(active_indices.size)
        ):
            raise RuntimeError("newborn proposal strengths are not zero")
        for name in ("strip", "birth_step", "ring_id", "sheet_id"):
            if not np.array_equal(
                getattr(candidate, name)[:old_ring_count], getattr(before, name)
            ):
                raise RuntimeError(f"Eq. 7 remeshing changed old {name}")
        geometry_changed = candidate.rings[:old_ring_count] != before.rings
        if np.any(geometry_changed & ~allowed_geometry_change):
            raise RuntimeError("Eq. 7 changed geometry outside the newest front edge")
        velocity_before = before.previous_vertex_velocity
        velocity_after = candidate.previous_vertex_velocity[:old_ring_count]
        velocity_change = ~(
            (velocity_after == velocity_before)
            | (np.isnan(velocity_after) & np.isnan(velocity_before))
        )
        if np.any(velocity_change & ~allowed_geometry_change):
            raise RuntimeError("Eq. 7 changed unrelated Eq. 24 history")

        return PteraMaterialLEVProposal(
            owner=self,
            candidate_shadow=candidate,
            step=step_value,
            base_version=self._version,
            active=active_mask,
            created_ids=report.created_ids,
            active_strip_indices=active_indices,
            old_ring_count=old_ring_count,
        )

    def _commit_proposal(
        self, proposal: PteraMaterialLEVProposal
    ) -> PteraMaterialLEVSnapshot:
        if proposal._owner is not self:
            raise RuntimeError("proposal belongs to a different material state")
        if proposal.base_version != self._version:
            raise RuntimeError("proposal is stale because material state changed")
        if proposal.step != self._last_committed_step + 1:
            raise RuntimeError("proposal no longer matches the next material step")

        self._shadow = proposal._candidate_shadow
        self._last_committed_step = proposal.step
        self._active_last_step = proposal._active.copy()
        if self._shadow.rings.shape[0] == 0:
            self._last_convected_step = proposal.step
        self._version += 1
        return self.snapshot()

    def _commit_convected_proposal(
        self,
        proposal: PteraMaterialLEVProposal,
        candidate_shadow: HiratoSheetShadow,
    ) -> PteraMaterialLEVSnapshot:
        """Publish a fully validated shed-plus-convection transaction."""

        if proposal._owner is not self:
            raise RuntimeError("proposal belongs to a different material state")
        if proposal.base_version != self._version:
            raise RuntimeError("proposal is stale because material state changed")
        if proposal.step != self._last_committed_step + 1:
            raise RuntimeError("proposal no longer matches the next material step")
        self._shadow = candidate_shadow
        self._last_committed_step = proposal.step
        self._last_convected_step = proposal.step
        self._active_last_step = proposal._active.copy()
        self._version += 1
        return self.snapshot()

    def convect_eq24(
        self,
        vertex_velocity_ptera_m_s: Any,
        *,
        dt_s: float,
        step: int,
    ) -> PteraEq24ConvectionReport:
        """Convect all material vertices using Ptera-ordered local velocity."""

        step_value = _step(step)
        if step_value != self._last_committed_step:
            raise HiratoEquationError(
                "Eq. 24 convection step must equal the latest committed shed step"
            )
        if step_value <= self._last_convected_step:
            raise HiratoEquationError(
                "material state was already convected at this step"
            )
        dt = _positive_float(dt_s, name="dt_s")
        velocity_ptera = _ring_array(
            vertex_velocity_ptera_m_s,
            name="vertex_velocity_ptera_m_s",
        )
        if velocity_ptera.shape != self._shadow.rings.shape:
            raise HiratoEquationError(
                "vertex_velocity_ptera_m_s must match the material-ring shape"
            )

        candidate = copy.deepcopy(self._shadow)
        gamma_before = candidate.gamma.copy()
        identity_before = (
            candidate.strip.copy(),
            candidate.birth_step.copy(),
            candidate.ring_id.copy(),
            candidate.sheet_id.copy(),
        )
        report = candidate.convect_eq24(
            _ptera_to_shadow_vertex_values(
                velocity_ptera, name="vertex_velocity_ptera_m_s"
            ),
            dt,
            step=step_value,
        )
        if not _bitwise_equal(candidate.gamma, gamma_before):
            raise RuntimeError("Eq. 24 convection changed material circulation")
        for current, before in zip(
            (
                candidate.strip,
                candidate.birth_step,
                candidate.ring_id,
                candidate.sheet_id,
            ),
            identity_before,
            strict=True,
        ):
            if not np.array_equal(current, before):
                raise RuntimeError("Eq. 24 convection changed material identity")

        self._shadow = candidate
        self._last_convected_step = step_value
        self._version += 1
        return PteraEq24ConvectionReport(
            step=report.step,
            bootstrap_vertex=report.bootstrap_vertex[
                :, PTERA_REVERSE_RING_TRAVERSAL
            ].copy(),
            displacement_gp1_m=_shadow_to_ptera_vertex_values(
                report.displacement, name="Eq24 displacement"
            ),
        )


__all__ = [
    "PTERA_REVERSE_RING_TRAVERSAL",
    "PteraEq24ConvectionReport",
    "PteraMaterialLEVProposal",
    "PteraMaterialLEVSnapshot",
    "PteraMaterialLEVView",
    "PteraNativeMaterialLEVState",
    "ptera_to_shadow_forward_rings",
    "shadow_forward_to_ptera_rings",
]
