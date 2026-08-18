"""Fail-closed, source-only Ramesh-LDVM contract for FluxV v5h.

The adapter runs the clean-room :class:`LDVM2D` source-parity mode and exports
only the quantities needed to create a newborn TE/LE vortex.  FluxV/Ptera
remains the sole aerodynamic-load owner.  The interface deliberately carries
enough geometry, dimensionalization, lineage, LESP, and Kelvin bookkeeping to
make every source record independently auditable.

Passing the local 174-LEV regression is necessary but not sufficient for a
canonical source.  Canonical promotion remains impossible here until a strict
D0 proof artifact is independently consumed by the later integration gate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import hmac
import json
from numbers import Integral, Real
import re
from typing import Any, Literal
import weakref

import numpy as np

from .ldvm_uvlm_correction import LDVMSectionSettings, LESPThreshold


SOURCE_INTERFACE_ID = "fluxv-v5h-dvm-source-only-v3"
SOURCE_PLACEMENT_SCHEMA_ID = "fluxv-v5h-dvm-source-placement-v3"
SOURCE_BACKEND_ID = (
    "platform.ldvm_fourier.LDVM2D-source-parity-clean-linear-provisional-tev-v3"
)
SOURCE_STATUS = "evaluated_source_only_noncanonical_d0_unconsumed"
DISABLED_STATUS = "not_evaluated_disabled"
CANONICAL_BLOCKER = (
    "strict D0 source-proof artifact has not been independently consumed by "
    "the D1 integration gate; local 174-LEV event parity alone is insufficient"
)
SOURCE_SOLVER = "clean_linear"
CIRCULATION_UNITS = "Gamma/(U_ref*c_ref)"
POSITION_UNITS = "(x/c_ref,z/c_ref)"
POSITION_FRAME = (
    "2D inertial LDVM world frame, x downstream and z up, origin at "
    "the section pivot at t*=0; birth position is pre-convection"
)
CIRCULATION_SIGN = (
    "positive is counter-clockwise in the x-z plane; this is the "
    "negative of the author-Fortran clockwise strength"
)
TEV_BIRTH_LAW = (
    "source-parity TEV: first source column is TE + 0.5*U*dt and "
    "its solved strength is zeroed before persistence; subsequent "
    "columns are TE + (previous newest TEV - TE)/3"
)
LEV_BIRTH_LAW = (
    "source-parity LEV: first/restart is LE + 0.5*v_LE*dt with "
    "v_LE induced by old wake plus the same-step provisional TEV; "
    "continuous shedding is LE + (previous newest LEV - LE)/3"
)
BIRTH_TIME_LAYER = (
    "birth coordinates are captured from the actual pre-convection "
    "constraint columns at t_n"
)
DIMENSIONALIZATION_LIMITATIONS = (
    "multiply circulation by U_ref*c_ref and position by c_ref; no "
    "span coordinate, edge endpoints, strip width, or 3D orientation "
    "is supplied, so a separately gated node-owned mapper is required"
)
BOTTOM_MODEL_PARITY = (
    "local strict source-parity regression reproduces the author event "
    "mask (174 LEVs, output rows 116..289) and first-TEV persistence; "
    "strengths use a clean linear solve rather than author Newton, and "
    "the independent D0 proof has not yet been consumed"
)
OWNERSHIP_SCOPE = "source quantities only; FluxV/Ptera is sole load owner"
OBSERVATION_ACCESS = "none"
TARGET_CASE_BRANCH = "none"
EVENT_CHAIN_DOMAIN = "fluxv-v5h-dvm-event-chain-v3"
ALLOWED_THRESHOLD_SOURCE_ROLES = frozenset(
    {
        "published_model_parameter",
        "published_source_input",
        "independent_non_target_calibration",
    }
)

LEVBirthMode = Literal["disabled", "none", "first", "continuous", "restart"]
VortexFamily = Literal["lev", "tev"]
PlacementMode = Literal["disabled", "inactive", "first", "continuous", "restart"]

_UNKNOWN_WORDS = frozenset(
    {
        "default",
        "na",
        "none",
        "null",
        "tbd",
        "todo",
        "unknown",
        "unset",
        "unspecified",
    }
)
_ZERO_CAMBER_NEGATIONS = re.compile(r"\b(?:non|not)\s+flat\b")


@dataclass(frozen=True, slots=True)
class DVMSourceProvenance:
    """Fixed physical identity, scaling, and ownership boundary."""

    interface_id: str
    backend_id: str
    physical_section_id: str
    physical_strip_id: str
    section_family: str
    reynolds: float
    lesp_critical: float
    threshold_source: str
    threshold_source_role: str
    delta_time_convective_nominal: float
    pivot_fraction_chord: float
    ndiv: int
    naterm: int
    resolved_core_radius_chord: float
    max_wake_steps: int
    geometry_identity: str
    geometry_hash_sha256: str
    geometry_role: str
    geometry_station_count: int
    circulation_units: str
    circulation_scale_u_times_c_m2_per_s: float
    position_units: str
    position_scale_chord_m: float
    position_frame: str
    circulation_sign: str
    tev_birth_law: str
    lev_birth_law: str
    birth_time_layer: str
    dimensionalization_limitations: str
    source_parity: bool
    source_solver: str
    canonical: bool
    canonical_blocker: str
    bottom_model_parity: str
    ownership_scope: str
    observation_access: str
    target_case_branch: str


@dataclass(frozen=True, slots=True)
class DVMSourceLineage:
    """Caller-owned strip identity and backend-history ownership."""

    physical_section_id: str
    physical_strip_id: str
    section_lineage_id: str
    source_step_index: int | None
    parent_state_step_index: int
    newborn_tev_source_id: str | None
    newborn_lev_source_id: str | None
    newborn_tev_role: str
    newborn_lev_role: str
    persistent_tev_history_role: str
    persistent_lev_history_role: str
    persistent_tev_count_before: int
    persistent_tev_count_after: int
    persistent_lev_count_before: int
    persistent_lev_count_after: int
    persistent_history_exported: bool


@dataclass(frozen=True, slots=True)
class DVMSourcePlacement:
    """Attested two-dimensional placement evidence for one newborn family.

    The absolute birth point and edge anchor are sampled at the same backend
    time layer.  A first/restarted placement exports the relative birth
    velocity that reproduces its half-step displacement.  A continuous
    placement instead exports the previous newborn's current backend point as
    audit evidence only; it is never permission to use that point for 3-D
    topology.
    """

    schema_id: str
    vortex_family: VortexFamily
    placement_mode: PlacementMode
    edge_anchor_position_over_chord_backend_world: tuple[float, float] | None
    birth_position_over_chord_backend_world: tuple[float, float] | None
    birth_displacement_from_edge_over_chord_backend_world: (tuple[float, float] | None)
    q_birth_over_u_backend_world: tuple[float, float] | None
    q_kinematic_over_u_backend_world: tuple[float, float] | None
    q_old_wake_over_u_backend_world: tuple[float, float] | None
    q_provisional_tev_over_u_backend_world: tuple[float, float] | None
    continuous_parent_source_id: str | None
    continuous_parent_position_over_chord_backend_world: (tuple[float, float] | None)
    used_for_topology_eligible: bool


@dataclass(frozen=True, slots=True)
class DVMKelvinLedger:
    """Complete nondimensional circulation ledger for one solved step."""

    circulation_units: str
    gamma_bound_post: float
    gamma_old_tev_persisted: float
    gamma_old_lev_persisted: float
    gamma_deleted_before: float
    gamma_tev_new_te_only_provisional: float
    gamma_tev_new_solved: float
    gamma_tev_new_persisted: float
    gamma_lev_new_solved: float
    gamma_lev_new_persisted: float
    gamma_deleted_after: float
    gamma_deleted_delta: float
    gamma_tev_persisted_after: float
    gamma_lev_persisted_after: float
    tev_solved_to_persisted_delta: float
    first_tev_zeroed: bool
    kelvin_solve_residual: float
    persistence_residual: float


@dataclass(frozen=True, slots=True, weakref_slot=True)
class DVMSourceEvent:
    """One source-only LDVM event; deliberately contains no load quantity."""

    enabled: bool
    status: str
    delta_time_convective: float | None
    a0_pre: float | None
    a0_post: float | None
    lesp_critical: float
    lesp_active: bool
    lesp_signed_target: float | None
    lesp_constraint_residual: float | None
    lev_birth_mode: LEVBirthMode
    restart: bool
    gamma_lev_new_over_u_c: float
    gamma_tev_new_solved_over_u_c: float
    gamma_tev_new_persisted_over_u_c: float
    lev_placement: DVMSourcePlacement
    tev_placement: DVMSourcePlacement
    lev_birth_position_over_chord_backend_world: tuple[float, float] | None
    tev_birth_position_over_chord_backend_world: tuple[float, float] | None
    kelvin_residual_over_u_c: float
    kelvin_ledger: DVMKelvinLedger | None
    lineage: DVMSourceLineage
    provenance: DVMSourceProvenance
    parent_event_manifest_sha256: str
    producer_manifest_sha256: str

    def manifest(self) -> dict[str, Any]:
        """Return a recursively JSON-compatible, source-only record."""

        return asdict(self)


# Only objects returned directly by this module's producer are registered.  The
# public digest makes accidental mutation visible; the weak identity registry
# prevents ``dataclasses.replace`` from manufacturing a trusted event merely by
# copying or recomputing that digest.  Serialized records therefore remain
# audit manifests, not trusted live source events.
_DIRECT_EVENT_REGISTRY: dict[
    int, tuple[weakref.ReferenceType[DVMSourceEvent], str]
] = {}


def _finite_scalar(name: str, value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real scalar")
    return result


def _strict_integer(name: str, value: object, *, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _strict_boolean(name: str, value: object) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be Boolean")
    return bool(value)


def _require_explicit_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be explicit")
    result = value.strip()
    words = set(re.findall(r"[a-z0-9]+", result.casefold()))
    if words & _UNKNOWN_WORDS:
        raise ValueError(f"{name} cannot contain unknown/default provenance")
    return result


def _finite_pair(name: str, value: object) -> tuple[float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name} must be a two-component finite real pair")
    return (
        _finite_scalar(f"{name}[0]", value[0]),
        _finite_scalar(f"{name}[1]", value[1]),
    )


def _pair_difference(
    end: tuple[float, float], start: tuple[float, float]
) -> tuple[float, float]:
    return (float(end[0] - start[0]), float(end[1] - start[1]))


def _sum_three_pairs(
    first: tuple[float, float],
    second: tuple[float, float],
    third: tuple[float, float],
) -> tuple[float, float]:
    return (
        float(first[0] + second[0] + third[0]),
        float(first[1] + second[1] + third[1]),
    )


def _direct_vatistas2_velocity(
    *,
    target: tuple[float, float],
    source_x: np.ndarray,
    source_y: np.ndarray,
    source_gamma: np.ndarray,
    core_radius: float,
) -> tuple[float, float]:
    """Independently evaluate the backend's two-dimensional direct kernel.

    The source adapter intentionally owns this small scalar-target
    recomputation.  It does not call ``LDVM2D._wcol`` or infer velocity from a
    stored birth displacement, so a coherent backend placement error cannot
    certify itself.  TE and LE histories are evaluated in separate calls by
    the producer to preserve the backend's family-wise summation order.
    """

    px, py = _finite_pair("direct-kernel target", target)
    vx = np.asarray(source_x, dtype=float)
    vy = np.asarray(source_y, dtype=float)
    gamma = np.asarray(source_gamma, dtype=float)
    if vx.ndim != 1 or vy.shape != vx.shape or gamma.shape != vx.shape:
        raise RuntimeError("old-wake direct-kernel arrays have inconsistent shape")
    if not (
        np.all(np.isfinite(vx))
        and np.all(np.isfinite(vy))
        and np.all(np.isfinite(gamma))
    ):
        raise RuntimeError("old-wake direct-kernel arrays must be finite")
    rc = _finite_scalar("direct-kernel core radius", core_radius)
    if rc <= 0.0:
        raise RuntimeError("direct-kernel core radius must be positive")
    if vx.size == 0:
        return (0.0, 0.0)
    rx = px - vx
    ry = py - vy
    r2 = rx * rx + ry * ry
    factor = gamma / (2.0 * np.pi) / np.sqrt(r2 * r2 + rc**4)
    q = (float(np.sum(factor * (-ry))), float(np.sum(factor * rx)))
    return _finite_pair("direct-kernel velocity", q)


def _disabled_placement(family: VortexFamily) -> DVMSourcePlacement:
    return DVMSourcePlacement(
        schema_id=SOURCE_PLACEMENT_SCHEMA_ID,
        vortex_family=family,
        placement_mode="disabled",
        edge_anchor_position_over_chord_backend_world=None,
        birth_position_over_chord_backend_world=None,
        birth_displacement_from_edge_over_chord_backend_world=None,
        q_birth_over_u_backend_world=None,
        q_kinematic_over_u_backend_world=None,
        q_old_wake_over_u_backend_world=None,
        q_provisional_tev_over_u_backend_world=None,
        continuous_parent_source_id=None,
        continuous_parent_position_over_chord_backend_world=None,
        used_for_topology_eligible=False,
    )


def _source_placement(
    *,
    family: VortexFamily,
    mode: PlacementMode,
    edge_anchor: object,
    birth_position: object | None,
    delta_time_convective: float,
    q_kinematic: object | None = None,
    q_old_wake: object | None = None,
    q_provisional_tev: object | None = None,
    continuous_parent_source_id: str | None = None,
    continuous_parent_position: object | None = None,
) -> DVMSourcePlacement:
    """Build placement evidence only from captured same-layer backend points."""

    edge = _finite_pair(f"{family} edge anchor", edge_anchor)
    if mode == "inactive":
        if birth_position is not None:
            raise RuntimeError("inactive placement unexpectedly carries a birth point")
        birth = None
        displacement = None
        q_birth = None
        q_kinematic_pair = None
        q_old_wake_pair = None
        q_provisional_tev_pair = None
    else:
        birth = _finite_pair(f"{family} birth position", birth_position)
        displacement = _pair_difference(birth, edge)
        if mode in {"first", "restart"}:
            q_kinematic_pair = _finite_pair(f"{family} q_kinematic", q_kinematic)
            q_old_wake_pair = _finite_pair(f"{family} q_old_wake", q_old_wake)
            q_provisional_tev_pair = _finite_pair(
                f"{family} q_provisional_tev", q_provisional_tev
            )
            q_birth = _sum_three_pairs(
                q_kinematic_pair,
                q_old_wake_pair,
                q_provisional_tev_pair,
            )
            reconstructed = (
                float(edge[0] + 0.5 * q_birth[0] * delta_time_convective),
                float(edge[1] + 0.5 * q_birth[1] * delta_time_convective),
            )
            if reconstructed != birth:
                raise RuntimeError(
                    f"{family} live first/restart placement disagrees with "
                    "independently recomputed velocity components"
                )
        else:
            if any(
                value is not None
                for value in (q_kinematic, q_old_wake, q_provisional_tev)
            ):
                raise RuntimeError(
                    f"{family} non-first placement carries velocity components"
                )
            q_birth = None
            q_kinematic_pair = None
            q_old_wake_pair = None
            q_provisional_tev_pair = None

    if mode == "continuous":
        parent_id = _require_explicit_text(
            f"{family} continuous parent source id",
            continuous_parent_source_id,
        )
        parent_position = _finite_pair(
            f"{family} continuous parent position", continuous_parent_position
        )
    else:
        if (
            continuous_parent_source_id is not None
            or continuous_parent_position is not None
        ):
            raise RuntimeError("non-continuous placement carries a continuous parent")
        parent_id = None
        parent_position = None

    return DVMSourcePlacement(
        schema_id=SOURCE_PLACEMENT_SCHEMA_ID,
        vortex_family=family,
        placement_mode=mode,
        edge_anchor_position_over_chord_backend_world=edge,
        birth_position_over_chord_backend_world=birth,
        birth_displacement_from_edge_over_chord_backend_world=displacement,
        q_birth_over_u_backend_world=q_birth,
        q_kinematic_over_u_backend_world=q_kinematic_pair,
        q_old_wake_over_u_backend_world=q_old_wake_pair,
        q_provisional_tev_over_u_backend_world=q_provisional_tev_pair,
        continuous_parent_source_id=parent_id,
        continuous_parent_position_over_chord_backend_world=parent_position,
        used_for_topology_eligible=(family == "lev" and mode in {"first", "restart"}),
    )


def _validate_placement(
    placement: object,
    *,
    family: VortexFamily,
    expected_mode: PlacementMode,
    enabled: bool,
    delta_time_convective: float | None,
    expected_birth_position: object | None,
    section_lineage_id: str,
    source_step_index: int | None,
) -> DVMSourcePlacement:
    if type(placement) is not DVMSourcePlacement:
        raise ValueError(f"{family} placement must be DVMSourcePlacement")
    if placement.schema_id != SOURCE_PLACEMENT_SCHEMA_ID:
        raise ValueError(f"{family} placement schema is not pinned")
    if placement.vortex_family != family:
        raise ValueError(f"{family} placement vortex family is inconsistent")
    if placement.placement_mode != expected_mode:
        raise ValueError(f"{family} placement mode is inconsistent")
    topology_eligible = _strict_boolean(
        f"{family} placement.used_for_topology_eligible",
        placement.used_for_topology_eligible,
    )
    expected_topology_eligible = family == "lev" and expected_mode in {
        "first",
        "restart",
    }
    if topology_eligible != expected_topology_eligible:
        raise ValueError(f"{family} placement topology eligibility is inconsistent")

    coordinate_fields = (
        placement.edge_anchor_position_over_chord_backend_world,
        placement.birth_position_over_chord_backend_world,
        placement.birth_displacement_from_edge_over_chord_backend_world,
        placement.q_birth_over_u_backend_world,
        placement.q_kinematic_over_u_backend_world,
        placement.q_old_wake_over_u_backend_world,
        placement.q_provisional_tev_over_u_backend_world,
        placement.continuous_parent_source_id,
        placement.continuous_parent_position_over_chord_backend_world,
    )
    if not enabled:
        if expected_mode != "disabled" or any(
            value is not None for value in coordinate_fields
        ):
            raise ValueError(f"disabled {family} placement carries evaluated data")
        return placement

    if expected_mode == "disabled":
        raise ValueError(f"enabled {family} placement cannot be disabled")
    edge = _finite_pair(
        f"{family} placement edge anchor",
        placement.edge_anchor_position_over_chord_backend_world,
    )
    if expected_mode == "inactive":
        if family != "lev" or any(
            value is not None
            for value in (
                placement.birth_position_over_chord_backend_world,
                placement.birth_displacement_from_edge_over_chord_backend_world,
                placement.q_birth_over_u_backend_world,
                placement.q_kinematic_over_u_backend_world,
                placement.q_old_wake_over_u_backend_world,
                placement.q_provisional_tev_over_u_backend_world,
                placement.continuous_parent_source_id,
                placement.continuous_parent_position_over_chord_backend_world,
            )
        ):
            raise ValueError("inactive LEV placement carries newborn data")
        if expected_birth_position is not None:
            raise ValueError("inactive LEV placement disagrees with legacy birth data")
        return placement

    birth = _finite_pair(
        f"{family} placement birth position",
        placement.birth_position_over_chord_backend_world,
    )
    expected_birth = _finite_pair(
        f"legacy {family} birth position", expected_birth_position
    )
    if birth != expected_birth:
        raise ValueError(f"{family} placement and legacy birth positions disagree")
    displacement = _finite_pair(
        f"{family} placement birth displacement",
        placement.birth_displacement_from_edge_over_chord_backend_world,
    )
    if displacement != _pair_difference(birth, edge):
        raise ValueError(f"{family} placement displacement is not reproducible")

    if expected_mode in {"first", "restart"}:
        if delta_time_convective is None:
            raise ValueError(f"{family} first/restart placement has no time step")
        q_birth = _finite_pair(
            f"{family} placement q_birth",
            placement.q_birth_over_u_backend_world,
        )
        q_kinematic = _finite_pair(
            f"{family} placement q_kinematic",
            placement.q_kinematic_over_u_backend_world,
        )
        q_old_wake = _finite_pair(
            f"{family} placement q_old_wake",
            placement.q_old_wake_over_u_backend_world,
        )
        q_provisional_tev = _finite_pair(
            f"{family} placement q_provisional_tev",
            placement.q_provisional_tev_over_u_backend_world,
        )
        expected_q_birth = _sum_three_pairs(
            q_kinematic,
            q_old_wake,
            q_provisional_tev,
        )
        if q_birth != expected_q_birth:
            raise ValueError(
                f"{family} first/restart q_birth is not the exact component sum"
            )
        reconstructed = (
            float(edge[0] + 0.5 * q_birth[0] * delta_time_convective),
            float(edge[1] + 0.5 * q_birth[1] * delta_time_convective),
        )
        if reconstructed != birth:
            raise ValueError(
                f"{family} first/restart half-step placement is not reproducible"
            )
        if (
            placement.continuous_parent_source_id is not None
            or placement.continuous_parent_position_over_chord_backend_world is not None
        ):
            raise ValueError(f"{family} first/restart placement carries a parent")
        return placement

    if expected_mode != "continuous":
        raise ValueError(f"{family} placement mode is unsupported")
    if any(
        value is not None
        for value in (
            placement.q_birth_over_u_backend_world,
            placement.q_kinematic_over_u_backend_world,
            placement.q_old_wake_over_u_backend_world,
            placement.q_provisional_tev_over_u_backend_world,
        )
    ):
        raise ValueError(
            f"continuous {family} placement cannot carry velocity evidence"
        )
    if source_step_index is None or source_step_index < 2:
        raise ValueError(f"continuous {family} placement has no prior source step")
    expected_parent_id = (
        f"{section_lineage_id}:step:{source_step_index - 1}:{family}-newborn"
    )
    if placement.continuous_parent_source_id != expected_parent_id:
        raise ValueError(f"continuous {family} parent source identity is inconsistent")
    parent = _finite_pair(
        f"continuous {family} parent position",
        placement.continuous_parent_position_over_chord_backend_world,
    )
    reconstructed = (
        float(edge[0] + (parent[0] - edge[0]) / 3.0),
        float(edge[1] + (parent[1] - edge[1]) / 3.0),
    )
    if reconstructed != birth:
        raise ValueError(f"continuous {family} one-third placement is not reproducible")
    if topology_eligible:
        raise ValueError(f"continuous {family} placement cannot own topology")
    return placement


def _event_manifest_digest(event: DVMSourceEvent) -> str:
    payload = asdict(event)
    payload.pop("producer_manifest_sha256")
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(b"fluxv-v5h-direct-dvm-event-v3\0" + encoded).hexdigest()


def _event_chain_genesis_digest(section_lineage_id: str) -> str:
    """Return the deterministic parent digest for a section's first event."""

    encoded = json.dumps(
        [EVENT_CHAIN_DOMAIN, section_lineage_id],
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _section_lineage_id_from_provenance(provenance: DVMSourceProvenance) -> str:
    threshold_manifest = {
        "value": provenance.lesp_critical,
        "section_family": provenance.section_family,
        "reynolds": provenance.reynolds,
        "source": provenance.threshold_source,
        "source_role": provenance.threshold_source_role,
    }
    payload = {
        "provenance": asdict(provenance),
        "threshold": threshold_manifest,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return f"dvm-section-{sha256(encoded).hexdigest()[:20]}"


def _validate_source_provenance(provenance: object) -> DVMSourceProvenance:
    if type(provenance) is not DVMSourceProvenance:
        raise ValueError("event provenance must be DVMSourceProvenance")

    exact_fields = {
        "interface_id": SOURCE_INTERFACE_ID,
        "backend_id": SOURCE_BACKEND_ID,
        "circulation_units": CIRCULATION_UNITS,
        "position_units": POSITION_UNITS,
        "position_frame": POSITION_FRAME,
        "circulation_sign": CIRCULATION_SIGN,
        "tev_birth_law": TEV_BIRTH_LAW,
        "lev_birth_law": LEV_BIRTH_LAW,
        "birth_time_layer": BIRTH_TIME_LAYER,
        "dimensionalization_limitations": DIMENSIONALIZATION_LIMITATIONS,
        "source_solver": SOURCE_SOLVER,
        "canonical_blocker": CANONICAL_BLOCKER,
        "bottom_model_parity": BOTTOM_MODEL_PARITY,
        "ownership_scope": OWNERSHIP_SCOPE,
        "observation_access": OBSERVATION_ACCESS,
        "target_case_branch": TARGET_CASE_BRANCH,
    }
    for name, expected in exact_fields.items():
        if getattr(provenance, name) != expected:
            raise ValueError(
                f"DVM provenance {name} does not match the pinned contract"
            )
    if (
        _strict_boolean("provenance.source_parity", provenance.source_parity)
        is not True
    ):
        raise ValueError("DVM provenance must pin source_parity=True")
    if _strict_boolean("provenance.canonical", provenance.canonical) is not False:
        raise ValueError("DVM provenance must remain canonical=False")

    section_id = _require_explicit_text(
        "provenance.physical_section_id", provenance.physical_section_id
    )
    strip_id = _require_explicit_text(
        "provenance.physical_strip_id", provenance.physical_strip_id
    )
    if section_id == strip_id:
        raise ValueError("provenance physical section and strip IDs must be distinct")
    _require_explicit_text("provenance.section_family", provenance.section_family)
    _require_explicit_text("provenance.threshold_source", provenance.threshold_source)
    threshold_role = _require_explicit_text(
        "provenance.threshold_source_role", provenance.threshold_source_role
    )
    if threshold_role not in ALLOWED_THRESHOLD_SOURCE_ROLES:
        raise ValueError("provenance threshold source role is not source-safe")

    reynolds = _finite_scalar("provenance.reynolds", provenance.reynolds)
    critical = _finite_scalar("provenance.lesp_critical", provenance.lesp_critical)
    nominal_dt = _finite_scalar(
        "provenance.delta_time_convective_nominal",
        provenance.delta_time_convective_nominal,
    )
    core = _finite_scalar(
        "provenance.resolved_core_radius_chord",
        provenance.resolved_core_radius_chord,
    )
    circulation_scale = _finite_scalar(
        "provenance.circulation_scale_u_times_c_m2_per_s",
        provenance.circulation_scale_u_times_c_m2_per_s,
    )
    position_scale = _finite_scalar(
        "provenance.position_scale_chord_m", provenance.position_scale_chord_m
    )
    if (
        min(reynolds, critical, nominal_dt, core, circulation_scale, position_scale)
        <= 0
    ):
        raise ValueError("DVM provenance positive physical scalars must be positive")
    pivot = _finite_scalar(
        "provenance.pivot_fraction_chord", provenance.pivot_fraction_chord
    )
    if not 0.0 <= pivot <= 1.0:
        raise ValueError("provenance pivot_fraction_chord must lie in [0, 1]")

    ndiv = _strict_integer("provenance.ndiv", provenance.ndiv, minimum=12)
    naterm = _strict_integer("provenance.naterm", provenance.naterm, minimum=3)
    _strict_integer("provenance.max_wake_steps", provenance.max_wake_steps, minimum=8)
    station_count = _strict_integer(
        "provenance.geometry_station_count",
        provenance.geometry_station_count,
        minimum=12,
    )
    if naterm >= ndiv or station_count != ndiv:
        raise ValueError("DVM provenance Fourier and geometry counts are inconsistent")
    _require_explicit_text("provenance.geometry_identity", provenance.geometry_identity)
    if provenance.geometry_role not in {
        "explicit_zero_camber_surrogate",
        "explicit_paired_camber_ordinate_and_slope",
    }:
        raise ValueError("DVM provenance geometry_role is not supported")
    if (
        not isinstance(provenance.geometry_hash_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", provenance.geometry_hash_sha256) is None
    ):
        raise ValueError("DVM provenance geometry hash must be lowercase SHA-256")
    return provenance


def _validate_event_semantics(event: object) -> DVMSourceEvent:
    if type(event) is not DVMSourceEvent:
        raise ValueError("event must be exactly DVMSourceEvent")
    enabled = _strict_boolean("event.enabled", event.enabled)
    active = _strict_boolean("event.lesp_active", event.lesp_active)
    restart = _strict_boolean("event.restart", event.restart)
    provenance = _validate_source_provenance(event.provenance)
    lineage = event.lineage
    if type(lineage) is not DVMSourceLineage:
        raise ValueError("event lineage must be DVMSourceLineage")
    if lineage.physical_section_id != provenance.physical_section_id:
        raise ValueError("lineage physical_section_id disagrees with provenance")
    if lineage.physical_strip_id != provenance.physical_strip_id:
        raise ValueError("lineage physical_strip_id disagrees with provenance")
    expected_lineage_id = _section_lineage_id_from_provenance(provenance)
    if lineage.section_lineage_id != expected_lineage_id:
        raise ValueError("section_lineage_id is not reproducible from provenance")
    parent_digest = event.parent_event_manifest_sha256
    if (
        not isinstance(parent_digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", parent_digest) is None
    ):
        raise ValueError("DVM parent-event digest is not a lowercase SHA-256")
    if (
        _strict_boolean(
            "lineage.persistent_history_exported", lineage.persistent_history_exported
        )
        is not False
    ):
        raise ValueError("persistent DVM history cannot be exported as newborn")
    if lineage.persistent_tev_history_role != (
        "backend-owned convected TE history; not re-exported as newborn"
    ) or lineage.persistent_lev_history_role != (
        "backend-owned convected LE history; not re-exported as newborn"
    ):
        raise ValueError("persistent-history lineage roles are not pinned")

    parent = _strict_integer(
        "lineage.parent_state_step_index",
        lineage.parent_state_step_index,
        minimum=0,
    )
    tev_before = _strict_integer(
        "lineage.persistent_tev_count_before",
        lineage.persistent_tev_count_before,
        minimum=0,
    )
    tev_after = _strict_integer(
        "lineage.persistent_tev_count_after",
        lineage.persistent_tev_count_after,
        minimum=0,
    )
    lev_before = _strict_integer(
        "lineage.persistent_lev_count_before",
        lineage.persistent_lev_count_before,
        minimum=0,
    )
    lev_after = _strict_integer(
        "lineage.persistent_lev_count_after",
        lineage.persistent_lev_count_after,
        minimum=0,
    )
    critical = _finite_scalar("event.lesp_critical", event.lesp_critical)
    if critical != provenance.lesp_critical:
        raise ValueError("event and provenance LESP critical values disagree")

    if not enabled:
        if event.status != DISABLED_STATUS:
            raise ValueError("disabled DVM event status is not pinned")
        if lineage.source_step_index is not None:
            raise ValueError("disabled DVM event cannot have a source step")
        if any(
            value is not None
            for value in (
                event.delta_time_convective,
                event.a0_pre,
                event.a0_post,
                event.lesp_signed_target,
                event.lesp_constraint_residual,
                event.lev_birth_position_over_chord_backend_world,
                event.tev_birth_position_over_chord_backend_world,
                event.kelvin_ledger,
                lineage.newborn_tev_source_id,
                lineage.newborn_lev_source_id,
            )
        ):
            raise ValueError("disabled DVM event carries evaluated source data")
        if active or restart or event.lev_birth_mode != "disabled":
            raise ValueError("disabled DVM event carries an active source mode")
        if any(
            value != 0.0
            for value in (
                event.gamma_lev_new_over_u_c,
                event.gamma_tev_new_solved_over_u_c,
                event.gamma_tev_new_persisted_over_u_c,
                event.kelvin_residual_over_u_c,
            )
        ):
            raise ValueError("disabled DVM event carries circulation")
        if tev_before != tev_after or lev_before != lev_after:
            raise ValueError("disabled DVM event changed persistent counts")
        if lineage.newborn_tev_role != "disabled_no_newborn_source" or (
            lineage.newborn_lev_role != "disabled_no_newborn_source"
        ):
            raise ValueError("disabled DVM event newborn roles are not pinned")
        _validate_placement(
            event.lev_placement,
            family="lev",
            expected_mode="disabled",
            enabled=False,
            delta_time_convective=None,
            expected_birth_position=None,
            section_lineage_id=expected_lineage_id,
            source_step_index=None,
        )
        _validate_placement(
            event.tev_placement,
            family="tev",
            expected_mode="disabled",
            enabled=False,
            delta_time_convective=None,
            expected_birth_position=None,
            section_lineage_id=expected_lineage_id,
            source_step_index=None,
        )
        return event

    if event.status != SOURCE_STATUS:
        raise ValueError("evaluated DVM source status is not pinned")
    source_step = _strict_integer(
        "lineage.source_step_index", lineage.source_step_index, minimum=1
    )
    if source_step == 1 and parent_digest != _event_chain_genesis_digest(
        expected_lineage_id
    ):
        raise ValueError("first DVM event does not reference its chain genesis")
    if parent != source_step - 1:
        raise ValueError("DVM parent/source step lineage is inconsistent")
    max_wake = int(provenance.max_wake_steps)
    if tev_before != min(source_step - 1, max_wake) or tev_after != min(
        source_step, max_wake
    ):
        raise ValueError("DVM TEV lineage counts do not match the source step")
    expected_lev_after = min(lev_before + int(active), max_wake)
    if lev_before > max_wake or lev_after != expected_lev_after:
        raise ValueError("DVM LEV lineage count transition is inconsistent")
    expected_prefix = f"{expected_lineage_id}:step:{source_step}"
    if lineage.newborn_tev_source_id != f"{expected_prefix}:tev-newborn":
        raise ValueError("newborn TEV source identity is not reproducible")

    dt = _finite_scalar("event.delta_time_convective", event.delta_time_convective)
    if dt <= 0.0:
        raise ValueError("event delta_time_convective must be positive")
    a0_pre = _finite_scalar("event.a0_pre", event.a0_pre)
    a0_post = _finite_scalar("event.a0_post", event.a0_post)
    target = _finite_scalar("event.lesp_signed_target", event.lesp_signed_target)
    residual = _finite_scalar(
        "event.lesp_constraint_residual", event.lesp_constraint_residual
    )
    expected_active = abs(a0_pre) > critical
    if active != expected_active:
        raise ValueError("LESP activity disagrees with the strict source threshold")
    expected_target = float(np.clip(a0_pre, -critical, critical))
    tolerance = (
        512.0 * np.finfo(float).eps * max(1.0, abs(a0_pre), abs(a0_post), critical)
    )
    if active:
        if target != expected_target:
            raise ValueError("active LESP target is not the exact signed threshold")
        if abs(residual - (a0_post - target)) > tolerance:
            raise ValueError("active LESP residual is not independently reproducible")
        if abs(residual) > tolerance:
            raise ValueError("active LESP constraint residual exceeds tolerance")
        expected_mode: LEVBirthMode
        if lev_before == 0:
            expected_mode = "first"
        elif restart:
            expected_mode = "restart"
        else:
            expected_mode = "continuous"
        if event.lev_birth_mode != expected_mode or restart != (
            expected_mode == "restart"
        ):
            raise ValueError("active LEV birth mode/restart lineage is inconsistent")
        if lineage.newborn_lev_source_id != f"{expected_prefix}:lev-newborn":
            raise ValueError("newborn LEV source identity is not reproducible")
        if lineage.newborn_lev_role != "lesp_gated_newborn_persisted_step_source":
            raise ValueError("active newborn LEV role is not pinned")
        _finite_pair(
            "LEV birth position", event.lev_birth_position_over_chord_backend_world
        )
    else:
        if not (
            a0_pre == a0_post
            and target == a0_pre
            and residual == 0.0
            and event.gamma_lev_new_over_u_c == 0.0
        ):
            raise ValueError("inactive LESP event is not the exact attached reduction")
        if restart or event.lev_birth_mode != "none":
            raise ValueError("inactive LESP event carries a birth/restart mode")
        if event.lev_birth_position_over_chord_backend_world is not None:
            raise ValueError("inactive LESP event carries an LEV birth point")
        if lineage.newborn_lev_source_id is not None:
            raise ValueError("inactive LESP event carries a newborn LEV identity")
        if lineage.newborn_lev_role != "inactive_no_newborn_source":
            raise ValueError("inactive newborn LEV role is not pinned")
    _finite_pair(
        "TEV birth position", event.tev_birth_position_over_chord_backend_world
    )
    lev_placement_mode: PlacementMode = (
        "inactive" if event.lev_birth_mode == "none" else event.lev_birth_mode
    )
    _validate_placement(
        event.lev_placement,
        family="lev",
        expected_mode=lev_placement_mode,
        enabled=True,
        delta_time_convective=dt,
        expected_birth_position=event.lev_birth_position_over_chord_backend_world,
        section_lineage_id=expected_lineage_id,
        source_step_index=source_step,
    )
    _validate_placement(
        event.tev_placement,
        family="tev",
        expected_mode=("first" if source_step == 1 else "continuous"),
        enabled=True,
        delta_time_convective=dt,
        expected_birth_position=event.tev_birth_position_over_chord_backend_world,
        section_lineage_id=expected_lineage_id,
        source_step_index=source_step,
    )

    ledger = event.kelvin_ledger
    if type(ledger) is not DVMKelvinLedger:
        raise ValueError("evaluated DVM event must carry a Kelvin ledger")
    if ledger.circulation_units != CIRCULATION_UNITS:
        raise ValueError("Kelvin ledger circulation units are not pinned")
    first_tev_zeroed = _strict_boolean(
        "ledger.first_tev_zeroed", ledger.first_tev_zeroed
    )
    if first_tev_zeroed != (source_step == 1):
        raise ValueError("first-TEV persistence flag disagrees with source lineage")
    ledger_values = {
        name: _finite_scalar(f"ledger.{name}", getattr(ledger, name))
        for name in (
            "gamma_bound_post",
            "gamma_old_tev_persisted",
            "gamma_old_lev_persisted",
            "gamma_deleted_before",
            "gamma_tev_new_te_only_provisional",
            "gamma_tev_new_solved",
            "gamma_tev_new_persisted",
            "gamma_lev_new_solved",
            "gamma_lev_new_persisted",
            "gamma_deleted_after",
            "gamma_deleted_delta",
            "gamma_tev_persisted_after",
            "gamma_lev_persisted_after",
            "tev_solved_to_persisted_delta",
            "kelvin_solve_residual",
            "persistence_residual",
        )
    }
    if (
        event.gamma_tev_new_solved_over_u_c != ledger_values["gamma_tev_new_solved"]
        or event.gamma_tev_new_persisted_over_u_c
        != ledger_values["gamma_tev_new_persisted"]
    ):
        raise ValueError("event and Kelvin-ledger TEV strengths disagree")
    if (
        not active
        and ledger_values["gamma_tev_new_te_only_provisional"]
        != ledger_values["gamma_tev_new_solved"]
    ):
        raise ValueError(
            "attached TE-only provisional and final TEV strengths must match"
        )
    if event.gamma_lev_new_over_u_c != ledger_values["gamma_lev_new_persisted"]:
        raise ValueError("event and Kelvin-ledger LEV strengths disagree")
    if (
        ledger_values["gamma_lev_new_solved"]
        != ledger_values["gamma_lev_new_persisted"]
    ):
        raise ValueError("solved and persisted LEV strengths must be identical")
    if first_tev_zeroed:
        if ledger_values["gamma_tev_new_persisted"] != 0.0:
            raise ValueError("first solved TEV must be zeroed before persistence")
    elif (
        ledger_values["gamma_tev_new_persisted"]
        != ledger_values["gamma_tev_new_solved"]
    ):
        raise ValueError("non-first solved and persisted TEV strengths must match")

    solve = (
        -ledger_values["gamma_bound_post"]
        + ledger_values["gamma_old_tev_persisted"]
        + ledger_values["gamma_old_lev_persisted"]
        + ledger_values["gamma_deleted_before"]
        + ledger_values["gamma_tev_new_solved"]
        + ledger_values["gamma_lev_new_solved"]
    )
    persistence = (
        ledger_values["gamma_tev_persisted_after"]
        + ledger_values["gamma_lev_persisted_after"]
        + ledger_values["gamma_deleted_after"]
        - (
            ledger_values["gamma_old_tev_persisted"]
            + ledger_values["gamma_old_lev_persisted"]
            + ledger_values["gamma_deleted_before"]
            + ledger_values["gamma_tev_new_persisted"]
            + ledger_values["gamma_lev_new_persisted"]
        )
    )
    ledger_scale = max(1.0, *(abs(value) for value in ledger_values.values()))
    ledger_tolerance = 1024.0 * np.finfo(float).eps * ledger_scale
    identities = (
        (solve, ledger_values["kelvin_solve_residual"], "Kelvin solve"),
        (persistence, ledger_values["persistence_residual"], "persistence"),
        (
            ledger_values["gamma_deleted_after"]
            - ledger_values["gamma_deleted_before"],
            ledger_values["gamma_deleted_delta"],
            "deleted-circulation delta",
        ),
        (
            ledger_values["gamma_tev_new_persisted"]
            - ledger_values["gamma_tev_new_solved"],
            ledger_values["tev_solved_to_persisted_delta"],
            "TEV solved-to-persisted delta",
        ),
    )
    for recomputed, saved, name in identities:
        if abs(recomputed - saved) > ledger_tolerance:
            raise ValueError(f"{name} ledger is not independently reproducible")
    if abs(ledger_values["kelvin_solve_residual"]) > ledger_tolerance:
        raise ValueError("Kelvin solve residual exceeds source tolerance")
    if abs(ledger_values["persistence_residual"]) > ledger_tolerance:
        raise ValueError("persistence residual exceeds source tolerance")
    if event.kelvin_residual_over_u_c != ledger_values["kelvin_solve_residual"]:
        raise ValueError("event and detailed Kelvin residuals disagree")
    expected_tev_role = (
        "constraint_column_solved_then_zeroed_before_persistence"
        if first_tev_zeroed
        else "coupled_newest_persisted_step_source"
    )
    if lineage.newborn_tev_role != expected_tev_role:
        raise ValueError("newborn TEV role disagrees with persistence lineage")
    return event


def _attest_direct_event(event: DVMSourceEvent) -> DVMSourceEvent:
    _validate_event_semantics(event)
    digest = _event_manifest_digest(event)
    object.__setattr__(event, "producer_manifest_sha256", digest)
    event_id = id(event)

    def discard(reference: weakref.ReferenceType[DVMSourceEvent]) -> None:
        existing = _DIRECT_EVENT_REGISTRY.get(event_id)
        if existing is not None and existing[0] is reference:
            _DIRECT_EVENT_REGISTRY.pop(event_id, None)

    reference = weakref.ref(event, discard)
    _DIRECT_EVENT_REGISTRY[event_id] = (reference, digest)
    return event


def validate_dvm_source_event(event: object) -> DVMSourceEvent:
    """Validate a live, directly produced source event and return it.

    A JSON manifest is useful evidence but is intentionally insufficient as a
    live handoff.  Trust additionally requires object identity registration by
    :class:`V5hDVMSource`, so copying with ``dataclasses.replace`` fails closed.
    """

    validated = _validate_event_semantics(event)
    attestation = validated.producer_manifest_sha256
    if (
        not isinstance(attestation, str)
        or re.fullmatch(r"[0-9a-f]{64}", attestation) is None
    ):
        raise ValueError("DVM event producer attestation is not a lowercase SHA-256")
    recomputed = _event_manifest_digest(validated)
    if not hmac.compare_digest(attestation, recomputed):
        raise ValueError("DVM event producer manifest digest does not match its fields")
    registered = _DIRECT_EVENT_REGISTRY.get(id(validated))
    if (
        registered is None
        or registered[0]() is not validated
        or not hmac.compare_digest(registered[1], recomputed)
    ):
        raise ValueError("DVM event is not a directly produced live source object")
    return validated


def _is_explicit_zero_camber_description(value: str) -> bool:
    """Accept an affirmative flat/zero-camber description, never a substring."""

    normalized = re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()
    words = set(normalized.split())
    if (
        _ZERO_CAMBER_NEGATIONS.search(normalized)
        or words.intersection({"not", "non"})
        or "cambered" in words
    ):
        return False
    return {"flat", "plate"}.issubset(words) or {"zero", "camber"}.issubset(words)


def _validate_threshold(threshold: object) -> LESPThreshold:
    if not isinstance(threshold, LESPThreshold):
        raise ValueError("threshold must be an explicit LESPThreshold")
    _require_explicit_text("section_family", threshold.section_family)
    _require_explicit_text("threshold source", threshold.source)
    role = _require_explicit_text("threshold source_role", threshold.source_role)
    if role not in ALLOWED_THRESHOLD_SOURCE_ROLES:
        raise ValueError(
            "threshold source_role is not source-safe; expected one of "
            f"{sorted(ALLOWED_THRESHOLD_SOURCE_ROLES)}"
        )
    value = _finite_scalar("LESP threshold", threshold.value)
    reynolds = _finite_scalar("Reynolds number", threshold.reynolds)
    if value <= 0.0 or reynolds <= 0.0:
        raise ValueError("LESP threshold and Reynolds number must be positive")
    return threshold


def _validate_settings(settings: object) -> tuple[int, int, int, float, float]:
    if not isinstance(settings, LDVMSectionSettings):
        raise ValueError("settings must be LDVMSectionSettings")
    ndiv = _strict_integer("settings.ndiv", settings.ndiv, minimum=12)
    naterm = _strict_integer("settings.naterm", settings.naterm, minimum=3)
    max_wake = _strict_integer(
        "settings.max_wake_steps", settings.max_wake_steps, minimum=8
    )
    if naterm >= ndiv:
        raise ValueError("settings.naterm must be smaller than settings.ndiv")
    ratio = _finite_scalar(
        "settings.core_radius_time_step_ratio",
        settings.core_radius_time_step_ratio,
    )
    attached = _finite_scalar(
        "settings.attached_lesp_critical", settings.attached_lesp_critical
    )
    if ratio <= 0.0 or attached <= 0.0:
        raise ValueError("settings radii and thresholds must be positive")
    if settings.core_radius_chord is not None:
        core = _finite_scalar("settings.core_radius_chord", settings.core_radius_chord)
        if core <= 0.0:
            raise ValueError("settings.core_radius_chord must be positive")
    else:
        core = np.nan
    return ndiv, naterm, max_wake, ratio, core


def _geometry_arrays(
    *,
    ndiv: int,
    section_family: str,
    geometry_identity: object,
    zero_camber_surrogate: object,
    camber_z_over_chord: object | None,
    camber_slope: object | None,
) -> tuple[str, str, str, np.ndarray, np.ndarray]:
    identity = _require_explicit_text("geometry_identity", geometry_identity)
    zero_surrogate = _strict_boolean("zero_camber_surrogate", zero_camber_surrogate)
    if zero_surrogate:
        if camber_z_over_chord is not None or camber_slope is not None:
            raise ValueError("zero-camber surrogate cannot also provide camber arrays")
        if not _is_explicit_zero_camber_description(identity):
            raise ValueError(
                "zero-camber geometry_identity must affirmatively say flat plate "
                "or zero camber"
            )
        if not _is_explicit_zero_camber_description(section_family):
            raise ValueError(
                "zero-camber surrogate cannot claim a non-flat section_family"
            )
        zc = np.zeros(ndiv, dtype=float)
        dzc = np.zeros(ndiv, dtype=float)
        role = "explicit_zero_camber_surrogate"
    else:
        if camber_z_over_chord is None or camber_slope is None:
            raise ValueError("paired camber_z_over_chord and camber_slope are required")
        validated_arrays: list[np.ndarray] = []
        for name, value in (
            ("camber_z_over_chord", camber_z_over_chord),
            ("camber_slope", camber_slope),
        ):
            raw = np.asarray(value, dtype=object)
            if raw.shape != (ndiv,):
                raise ValueError(f"camber arrays must both have shape {(ndiv,)}")
            if any(
                isinstance(item, (bool, np.bool_)) or not isinstance(item, Real)
                for item in raw.flat
            ):
                raise ValueError(f"{name} must contain finite real values")
            validated_arrays.append(np.asarray(raw, dtype=float))
        zc, dzc = validated_arrays
        if not np.all(np.isfinite(zc)) or not np.all(np.isfinite(dzc)):
            raise ValueError("camber arrays must contain finite real values")
        zc = zc.copy()
        dzc = dzc.copy()
        role = "explicit_paired_camber_ordinate_and_slope"

    digest = sha256()
    digest.update(b"fluxv-v5h-camber-pair-v1\0")
    digest.update(np.asarray([ndiv], dtype="<i8").tobytes())
    digest.update(np.asarray(zc, dtype="<f8").tobytes())
    digest.update(np.asarray(dzc, dtype="<f8").tobytes())
    return identity, digest.hexdigest(), role, zc, dzc


def _build_capture_model(**kwargs: Any) -> Any:
    """Build LDVM2D and capture the actual pre-convection birth columns."""

    from ldvm_fourier import LDVM2D

    class _BirthCaptureLDVM2D(LDVM2D):
        def __init__(self, **model_kwargs: Any) -> None:
            self.v5h_birth_columns: list[tuple[float, float]] = []
            self.v5h_edge_anchors: dict[str, tuple[float, float]] = {}
            self._v5h_capture_active = False
            super().__init__(**model_kwargs)

        def _wcol(self, px: float, py: float) -> np.ndarray:
            if self._v5h_capture_active:
                point = (_finite_scalar("birth x", px), _finite_scalar("birth y", py))
                self.v5h_birth_columns.append(point)
            return super()._wcol(px, py)

        def step(
            self,
            alpha: float,
            dalpha: float,
            hdot: float = 0.0,
            *,
            dt_i: float | None = None,
        ) -> Any:
            if self._v5h_capture_active:
                raise RuntimeError("nested LDVM source step is not supported")
            self.v5h_birth_columns = []
            self.v5h_edge_anchors = {}
            self._v5h_capture_active = True
            try:
                result = super().step(alpha, dalpha, hdot, dt_i=dt_i)
                # Wake convection does not alter the section pose.  Therefore
                # these direct calls to the live backend geometry are at the
                # exact same layer as the pre-convection constraint columns.
                self.v5h_edge_anchors = {
                    "lev": _finite_pair("live LE edge anchor", self._world(0.0)),
                    "tev": _finite_pair("live TE edge anchor", self._world(self.c)),
                }
                return result
            finally:
                self._v5h_capture_active = False

    return _BirthCaptureLDVM2D(**kwargs)


class V5hDVMSource:
    """Stateful source-only view of one explicitly identified physical strip."""

    def __init__(
        self,
        *,
        physical_section_id: str,
        physical_strip_id: str,
        geometry_identity: str,
        reference_speed_m_per_s: float,
        reference_chord_m: float,
        zero_camber_surrogate: bool,
        delta_time_convective: float,
        pivot_fraction_chord: float,
        threshold: LESPThreshold,
        settings: LDVMSectionSettings = LDVMSectionSettings(),
        camber_z_over_chord: object | None = None,
        camber_slope: object | None = None,
    ) -> None:
        threshold = _validate_threshold(threshold)
        physical_section = _require_explicit_text(
            "physical_section_id", physical_section_id
        )
        physical_strip = _require_explicit_text("physical_strip_id", physical_strip_id)
        if physical_section == physical_strip:
            raise ValueError("physical section and strip IDs must be distinct")
        dt = _finite_scalar("delta_time_convective", delta_time_convective)
        pivot = _finite_scalar("pivot_fraction_chord", pivot_fraction_chord)
        speed = _finite_scalar("reference_speed_m_per_s", reference_speed_m_per_s)
        chord = _finite_scalar("reference_chord_m", reference_chord_m)
        if dt <= 0.0 or speed <= 0.0 or chord <= 0.0:
            raise ValueError("time step, reference speed, and chord must be positive")
        if not 0.0 <= pivot <= 1.0:
            raise ValueError("pivot_fraction_chord must lie in [0, 1]")

        ndiv, naterm, max_wake, core_ratio, explicit_core = _validate_settings(settings)
        resolved_core_raw = (
            float(explicit_core) if np.isfinite(explicit_core) else core_ratio * dt
        )
        resolved_core = _finite_scalar("resolved core radius", resolved_core_raw)
        if resolved_core <= 0.0:
            raise ValueError("resolved core radius must be positive")
        circulation_scale = _finite_scalar(
            "circulation dimensionalization scale", speed * chord
        )
        if circulation_scale <= 0.0:
            raise ValueError("circulation dimensionalization scale must be positive")
        geometry, geometry_hash, geometry_role, zc, dzc = _geometry_arrays(
            ndiv=ndiv,
            section_family=threshold.section_family.strip(),
            geometry_identity=geometry_identity,
            zero_camber_surrogate=zero_camber_surrogate,
            camber_z_over_chord=camber_z_over_chord,
            camber_slope=camber_slope,
        )

        self._model = _build_capture_model(
            U=1.0,
            c=1.0,
            ndiv=ndiv,
            naterm=naterm,
            dt=dt,
            rho=1.0,
            lesp_crit=float(threshold.value),
            camber_m=0.0,
            pivot_xc=pivot,
            core_rc=resolved_core,
            max_wake=max_wake,
            source_parity=True,
            camber_z=zc,
            camber_slope=dzc,
        )
        self._threshold = threshold
        self._last_step_active = False
        self._ever_lev_shed = False
        self._poisoned = False
        self._nominal_dt = dt
        self._provenance = DVMSourceProvenance(
            interface_id=SOURCE_INTERFACE_ID,
            backend_id=SOURCE_BACKEND_ID,
            physical_section_id=physical_section,
            physical_strip_id=physical_strip,
            section_family=threshold.section_family.strip(),
            reynolds=float(threshold.reynolds),
            lesp_critical=float(threshold.value),
            threshold_source=threshold.source.strip(),
            threshold_source_role=threshold.source_role.strip(),
            delta_time_convective_nominal=dt,
            pivot_fraction_chord=pivot,
            ndiv=ndiv,
            naterm=naterm,
            resolved_core_radius_chord=resolved_core,
            max_wake_steps=max_wake,
            geometry_identity=geometry,
            geometry_hash_sha256=geometry_hash,
            geometry_role=geometry_role,
            geometry_station_count=ndiv,
            circulation_units=CIRCULATION_UNITS,
            circulation_scale_u_times_c_m2_per_s=circulation_scale,
            position_units=POSITION_UNITS,
            position_scale_chord_m=chord,
            position_frame=POSITION_FRAME,
            circulation_sign=CIRCULATION_SIGN,
            tev_birth_law=TEV_BIRTH_LAW,
            lev_birth_law=LEV_BIRTH_LAW,
            birth_time_layer=BIRTH_TIME_LAYER,
            dimensionalization_limitations=DIMENSIONALIZATION_LIMITATIONS,
            source_parity=True,
            source_solver=SOURCE_SOLVER,
            canonical=False,
            canonical_blocker=CANONICAL_BLOCKER,
            bottom_model_parity=BOTTOM_MODEL_PARITY,
            ownership_scope=OWNERSHIP_SCOPE,
            observation_access=OBSERVATION_ACCESS,
            target_case_branch=TARGET_CASE_BRANCH,
        )
        identity_payload = {
            "provenance": asdict(self._provenance),
            "threshold": threshold.manifest(),
        }
        encoded = json.dumps(
            identity_payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        self._section_lineage_id = f"dvm-section-{sha256(encoded).hexdigest()[:20]}"
        self._last_enabled_event_manifest_sha256 = _event_chain_genesis_digest(
            self._section_lineage_id
        )

    @property
    def step_count(self) -> int:
        """Number of enabled backend steps; disabled calls do not increment it."""

        return int(self._model.it)

    @property
    def provenance(self) -> DVMSourceProvenance:
        return self._provenance

    def _validate_live_backend_contract(self) -> None:
        """Reject mutable backend-constant drift before it can self-attest."""

        exact_scalars = {
            "U": 1.0,
            "c": 1.0,
            "xp": self._provenance.pivot_fraction_chord,
            "rc": self._provenance.resolved_core_radius_chord,
        }
        for name, expected in exact_scalars.items():
            actual = _finite_scalar(f"live backend {name}", getattr(self._model, name))
            if actual != expected:
                raise RuntimeError(f"live backend {name} drifted from provenance")
        exact_integers = {
            "ndiv": self._provenance.ndiv,
            "naterm": self._provenance.naterm,
            "max_wake": self._provenance.max_wake_steps,
        }
        for name, expected in exact_integers.items():
            actual = _strict_integer(
                f"live backend {name}", getattr(self._model, name), minimum=1
            )
            if actual != expected:
                raise RuntimeError(f"live backend {name} drifted from provenance")
        if (
            _strict_boolean("live backend source_parity", self._model.source_parity)
            is not True
        ):
            raise RuntimeError("live backend left source-parity mode")

        zc = np.asarray(self._model.zc, dtype=float)
        dzc = np.asarray(self._model.dzc, dtype=float)
        expected_shape = (self._provenance.geometry_station_count,)
        if (
            zc.shape != expected_shape
            or dzc.shape != expected_shape
            or not np.all(np.isfinite(zc))
            or not np.all(np.isfinite(dzc))
        ):
            raise RuntimeError("live backend camber geometry is invalid")
        digest = sha256()
        digest.update(b"fluxv-v5h-camber-pair-v1\0")
        digest.update(np.asarray([zc.size], dtype="<i8").tobytes())
        digest.update(np.asarray(zc, dtype="<f8").tobytes())
        digest.update(np.asarray(dzc, dtype="<f8").tobytes())
        if digest.hexdigest() != self._provenance.geometry_hash_sha256:
            raise RuntimeError("live backend camber geometry drifted from provenance")

    def _lineage(
        self,
        *,
        source_step_index: int | None,
        parent_step_index: int,
        tev_before: int,
        tev_after: int,
        lev_before: int,
        lev_after: int,
        lev_active: bool,
        first_tev_zeroed: bool,
    ) -> DVMSourceLineage:
        if source_step_index is None:
            tev_id = None
            lev_id = None
            tev_role = "disabled_no_newborn_source"
            lev_role = "disabled_no_newborn_source"
        else:
            prefix = f"{self._section_lineage_id}:step:{source_step_index}"
            tev_id = f"{prefix}:tev-newborn"
            lev_id = f"{prefix}:lev-newborn" if lev_active else None
            tev_role = (
                "constraint_column_solved_then_zeroed_before_persistence"
                if first_tev_zeroed
                else "coupled_newest_persisted_step_source"
            )
            lev_role = (
                "lesp_gated_newborn_persisted_step_source"
                if lev_active
                else "inactive_no_newborn_source"
            )
        return DVMSourceLineage(
            physical_section_id=self._provenance.physical_section_id,
            physical_strip_id=self._provenance.physical_strip_id,
            section_lineage_id=self._section_lineage_id,
            source_step_index=source_step_index,
            parent_state_step_index=parent_step_index,
            newborn_tev_source_id=tev_id,
            newborn_lev_source_id=lev_id,
            newborn_tev_role=tev_role,
            newborn_lev_role=lev_role,
            persistent_tev_history_role=(
                "backend-owned convected TE history; not re-exported as newborn"
            ),
            persistent_lev_history_role=(
                "backend-owned convected LE history; not re-exported as newborn"
            ),
            persistent_tev_count_before=tev_before,
            persistent_tev_count_after=tev_after,
            persistent_lev_count_before=lev_before,
            persistent_lev_count_after=lev_after,
            persistent_history_exported=False,
        )

    def _disabled_event(self) -> DVMSourceEvent:
        step = self.step_count
        tev_count = len(self._model.tg)
        lev_count = len(self._model.lg)
        event = DVMSourceEvent(
            enabled=False,
            status=DISABLED_STATUS,
            delta_time_convective=None,
            a0_pre=None,
            a0_post=None,
            lesp_critical=float(self._threshold.value),
            lesp_active=False,
            lesp_signed_target=None,
            lesp_constraint_residual=None,
            lev_birth_mode="disabled",
            restart=False,
            gamma_lev_new_over_u_c=0.0,
            gamma_tev_new_solved_over_u_c=0.0,
            gamma_tev_new_persisted_over_u_c=0.0,
            lev_placement=_disabled_placement("lev"),
            tev_placement=_disabled_placement("tev"),
            lev_birth_position_over_chord_backend_world=None,
            tev_birth_position_over_chord_backend_world=None,
            kelvin_residual_over_u_c=0.0,
            kelvin_ledger=None,
            lineage=self._lineage(
                source_step_index=None,
                parent_step_index=step,
                tev_before=tev_count,
                tev_after=tev_count,
                lev_before=lev_count,
                lev_after=lev_count,
                lev_active=False,
                first_tev_zeroed=False,
            ),
            provenance=self._provenance,
            parent_event_manifest_sha256=self._last_enabled_event_manifest_sha256,
            producer_manifest_sha256="",
        )
        return _attest_direct_event(event)

    def step(
        self,
        alpha_rad: object,
        alpha_rate_per_convective_time: object,
        heave_rate_over_u: object = 0.0,
        *,
        enabled: bool = True,
        delta_time_convective: object | None = None,
    ) -> DVMSourceEvent:
        """Advance one source step without exporting an LDVM load result."""

        enabled_value = _strict_boolean("enabled", enabled)
        if not enabled_value:
            return self._disabled_event()
        if self._poisoned:
            raise RuntimeError("DVM source state is fail-closed after a backend error")

        alpha = _finite_scalar("alpha_rad", alpha_rad)
        alpha_rate = _finite_scalar(
            "alpha_rate_per_convective_time", alpha_rate_per_convective_time
        )
        heave_rate = _finite_scalar("heave_rate_over_u", heave_rate_over_u)
        dt_step = (
            self._nominal_dt
            if delta_time_convective is None
            else _finite_scalar("delta_time_convective", delta_time_convective)
        )
        if dt_step <= 0.0:
            raise ValueError("delta_time_convective must be positive")

        self._validate_live_backend_contract()
        parent_step = self.step_count
        tev_before = len(self._model.tg)
        lev_before = len(self._model.lg)
        old_tev_x = np.asarray(self._model.tx, dtype=float).copy()
        old_tev_y = np.asarray(self._model.ty, dtype=float).copy()
        old_tev_gamma = np.asarray(self._model.tg, dtype=float).copy()
        old_lev_x = np.asarray(self._model.lx, dtype=float).copy()
        old_lev_y = np.asarray(self._model.ly, dtype=float).copy()
        old_lev_gamma = np.asarray(self._model.lg, dtype=float).copy()
        for family, x_values, y_values, gamma_values, expected_count in (
            ("TEV", old_tev_x, old_tev_y, old_tev_gamma, tev_before),
            ("LEV", old_lev_x, old_lev_y, old_lev_gamma, lev_before),
        ):
            if (
                x_values.shape != (expected_count,)
                or y_values.shape != (expected_count,)
                or gamma_values.shape != (expected_count,)
                or not np.all(np.isfinite(x_values))
                or not np.all(np.isfinite(y_values))
                or not np.all(np.isfinite(gamma_values))
            ):
                raise RuntimeError(f"pre-step {family} history snapshot is invalid")
        old_tev = _finite_scalar("old TEV circulation", np.sum(old_tev_gamma))
        old_lev = _finite_scalar("old LEV circulation", np.sum(old_lev_gamma))
        deleted_before = _finite_scalar(
            "deleted circulation before", self._model.gam_lost
        )
        previous_active = self._last_step_active
        ever_lev_before = self._ever_lev_shed

        try:
            tev_continuous_parent = (
                _finite_pair(
                    "pre-step newest TEV point",
                    (self._model.tx[-1], self._model.ty[-1]),
                )
                if tev_before > 0
                else None
            )
            lev_continuous_parent = (
                _finite_pair(
                    "pre-step newest LEV point",
                    (self._model.lx[-1], self._model.ly[-1]),
                )
                if previous_active and lev_before > 0
                else None
            )
            backend = self._model.step(alpha, alpha_rate, heave_rate, dt_i=dt_step)
            self._validate_live_backend_contract()
            required = {
                "lesp",
                "A0",
                "AF",
                "n_tev",
                "n_lev",
                "lesp_constraint_residual",
                "shed_lev",
                "dt_i",
                "source_parity",
                "tev_strength_te_only_provisional",
                "tev_strength_solved",
                "tev_strength_stored",
                "first_tev_zeroed",
            }
            if not isinstance(backend, dict) or not required.issubset(backend):
                raise RuntimeError("LDVM backend source diagnostics are incomplete")

            backend_source_parity = _strict_boolean(
                "backend source_parity", backend["source_parity"]
            )
            if not backend_source_parity:
                raise RuntimeError("LDVM backend left source-parity mode")
            backend_dt = _finite_scalar("backend dt_i", backend["dt_i"])
            if backend_dt != dt_step:
                raise RuntimeError("LDVM backend used a different source time step")

            a0_pre = _finite_scalar("A0_pre", backend["lesp"])
            a0_post = _finite_scalar("A0_post", backend["A0"])
            active = abs(a0_pre) > float(self._threshold.value)
            backend_active = _strict_boolean("backend shed_lev", backend["shed_lev"])
            if backend_active != active:
                raise RuntimeError("LDVM backend LESP event flag is inconsistent")
            signed_target = float(
                np.clip(
                    a0_pre,
                    -float(self._threshold.value),
                    float(self._threshold.value),
                )
            )
            residual = _finite_scalar(
                "LESP constraint residual", backend["lesp_constraint_residual"]
            )
            expected_residual = a0_post - signed_target
            residual_tolerance = (
                512.0 * np.finfo(float).eps * max(1.0, abs(a0_post), abs(signed_target))
            )
            if abs(residual - expected_residual) > residual_tolerance:
                raise RuntimeError(
                    "LDVM LESP residual cannot be independently reproduced"
                )

            columns = tuple(self._model.v5h_birth_columns)
            expected_columns = 2 if active else 1
            if len(columns) != expected_columns:
                raise RuntimeError(
                    "LDVM backend did not expose the expected actual birth columns"
                )
            anchors = self._model.v5h_edge_anchors
            if not isinstance(anchors, dict) or set(anchors) != {"lev", "tev"}:
                raise RuntimeError(
                    "LDVM backend did not expose same-layer live edge anchors"
                )
            lev_edge_anchor = _finite_pair("captured LE edge anchor", anchors["lev"])
            tev_edge_anchor = _finite_pair("captured TE edge anchor", anchors["tev"])
            live_lev_edge_anchor = _finite_pair(
                "live LE edge anchor", self._model._world(0.0)
            )
            live_tev_edge_anchor = _finite_pair(
                "live TE edge anchor", self._model._world(self._model.c)
            )
            if (
                lev_edge_anchor != live_lev_edge_anchor
                or tev_edge_anchor != live_tev_edge_anchor
            ):
                raise RuntimeError("captured edge anchor drifted from live geometry")

            tev_after = len(self._model.tg)
            lev_after = len(self._model.lg)
            backend_tev_count = _strict_integer(
                "backend n_tev", backend["n_tev"], minimum=0
            )
            backend_lev_count = _strict_integer(
                "backend n_lev", backend["n_lev"], minimum=0
            )
            expected_tev_after = min(tev_before + 1, int(self._model.max_wake))
            if tev_after != expected_tev_after or backend_tev_count != tev_after:
                raise RuntimeError(
                    "LDVM newest TE source/history state is inconsistent"
                )
            expected_lev_after = min(
                lev_before + int(active), int(self._model.max_wake)
            )
            if lev_after != expected_lev_after or backend_lev_count != lev_after:
                raise RuntimeError(
                    "LDVM newest LE source/history state is inconsistent"
                )

            gamma_tev_te_only_provisional = _finite_scalar(
                "Gamma_TEV_new_TE_only_provisional",
                backend["tev_strength_te_only_provisional"],
            )
            gamma_tev_solved = _finite_scalar(
                "Gamma_TEV_new_solved", backend["tev_strength_solved"]
            )
            gamma_tev_persisted = _finite_scalar(
                "Gamma_TEV_new_persisted", backend["tev_strength_stored"]
            )
            first_tev_zeroed = _strict_boolean(
                "backend first_tev_zeroed", backend["first_tev_zeroed"]
            )
            if first_tev_zeroed != (parent_step == 0):
                raise RuntimeError("LDVM first-TEV persistence flag is inconsistent")
            if self._model.tg[-1] != gamma_tev_persisted:
                raise RuntimeError("LDVM stored TEV diagnostic disagrees with history")
            if first_tev_zeroed and gamma_tev_persisted != 0.0:
                raise RuntimeError(
                    "LDVM first solved TEV was not zeroed for persistence"
                )
            if not first_tev_zeroed and gamma_tev_persisted != gamma_tev_solved:
                raise RuntimeError("LDVM solved and persisted TEV strengths diverged")
            gamma_lev = (
                _finite_scalar("Gamma_LEV_new", self._model.lg[-1]) if active else 0.0
            )
            if not active and gamma_tev_te_only_provisional != gamma_tev_solved:
                raise RuntimeError(
                    "attached TE-only provisional and final TEV strengths diverged"
                )

            af = np.asarray(backend["AF"], dtype=float)
            if af.shape != (4,) or not np.all(np.isfinite(af)):
                raise RuntimeError(
                    "LDVM backend post-constraint Fourier state is invalid"
                )
            gamma_bound_post = float(
                np.pi * self._model.c * self._model.U * (af[0] + 0.5 * af[1])
            )
            deleted_after = _finite_scalar(
                "deleted circulation after", self._model.gam_lost
            )
            persisted_tev_after = float(np.sum(np.asarray(self._model.tg, dtype=float)))
            persisted_lev_after = float(np.sum(np.asarray(self._model.lg, dtype=float)))
            kelvin_residual = float(
                -gamma_bound_post
                + old_tev
                + old_lev
                + deleted_before
                + gamma_tev_solved
                + gamma_lev
            )
            persistence_residual = float(
                persisted_tev_after
                + persisted_lev_after
                + deleted_after
                - (old_tev + old_lev + deleted_before + gamma_tev_persisted + gamma_lev)
            )
            scale = max(
                1.0,
                abs(gamma_bound_post),
                abs(old_tev),
                abs(old_lev),
                abs(deleted_before),
                abs(gamma_tev_te_only_provisional),
                abs(gamma_tev_solved),
                abs(gamma_lev),
                abs(deleted_after),
            )
            tolerance = 1024.0 * np.finfo(float).eps * scale
            if not np.isfinite(kelvin_residual) or abs(kelvin_residual) > tolerance:
                raise RuntimeError("LDVM newborn source fails its Kelvin solve ledger")
            if (
                not np.isfinite(persistence_residual)
                or abs(persistence_residual) > tolerance
            ):
                raise RuntimeError("LDVM wake trimming fails its persistence ledger")
            if not active and (a0_pre != a0_post or residual != 0.0):
                raise RuntimeError("LDVM no-shed step did not reduce to attached A0")
            if active and abs(residual) > residual_tolerance:
                raise RuntimeError("LDVM active step did not close the LESP constraint")

            first = active and not ever_lev_before
            restart = active and ever_lev_before and not previous_active
            if first:
                birth_mode: LEVBirthMode = "first"
            elif restart:
                birth_mode = "restart"
            elif active:
                birth_mode = "continuous"
            else:
                birth_mode = "none"
            self._last_step_active = active
            self._ever_lev_shed = ever_lev_before or active

            deleted_delta = deleted_after - deleted_before
            solved_to_persisted_delta = gamma_tev_persisted - gamma_tev_solved
            ledger = DVMKelvinLedger(
                circulation_units=self._provenance.circulation_units,
                gamma_bound_post=gamma_bound_post,
                gamma_old_tev_persisted=old_tev,
                gamma_old_lev_persisted=old_lev,
                gamma_deleted_before=deleted_before,
                gamma_tev_new_te_only_provisional=(gamma_tev_te_only_provisional),
                gamma_tev_new_solved=gamma_tev_solved,
                gamma_tev_new_persisted=gamma_tev_persisted,
                gamma_lev_new_solved=gamma_lev,
                gamma_lev_new_persisted=gamma_lev,
                gamma_deleted_after=deleted_after,
                gamma_deleted_delta=(
                    0.0 if deleted_delta == 0.0 else float(deleted_delta)
                ),
                gamma_tev_persisted_after=persisted_tev_after,
                gamma_lev_persisted_after=persisted_lev_after,
                tev_solved_to_persisted_delta=(
                    0.0
                    if solved_to_persisted_delta == 0.0
                    else float(solved_to_persisted_delta)
                ),
                first_tev_zeroed=first_tev_zeroed,
                kelvin_solve_residual=(
                    0.0 if kelvin_residual == 0.0 else kelvin_residual
                ),
                persistence_residual=(
                    0.0 if persistence_residual == 0.0 else persistence_residual
                ),
            )

            source_step = self.step_count
            tev_mode: PlacementMode = "first" if source_step == 1 else "continuous"
            tev_parent_id = (
                f"{self._section_lineage_id}:step:{source_step - 1}:tev-newborn"
                if tev_mode == "continuous"
                else None
            )
            tev_q_kinematic = (1.0, 0.0) if tev_mode == "first" else None
            tev_q_old_wake = (0.0, 0.0) if tev_mode == "first" else None
            tev_q_provisional = (0.0, 0.0) if tev_mode == "first" else None
            tev_placement = _source_placement(
                family="tev",
                mode=tev_mode,
                edge_anchor=tev_edge_anchor,
                birth_position=columns[0],
                delta_time_convective=dt_step,
                q_kinematic=tev_q_kinematic,
                q_old_wake=tev_q_old_wake,
                q_provisional_tev=tev_q_provisional,
                continuous_parent_source_id=tev_parent_id,
                continuous_parent_position=tev_continuous_parent,
            )
            lev_mode: PlacementMode = "inactive" if birth_mode == "none" else birth_mode
            lev_parent_id = (
                f"{self._section_lineage_id}:step:{source_step - 1}:lev-newborn"
                if lev_mode == "continuous"
                else None
            )
            lev_q_kinematic: tuple[float, float] | None = None
            lev_q_old_wake: tuple[float, float] | None = None
            lev_q_provisional: tuple[float, float] | None = None
            if lev_mode in {"first", "restart"}:
                # Recompute the author-ordered birth velocity from the frozen
                # pre-step wake, the live current-layer edge, and the frozen
                # TE-only provisional circulation.  The final coupled TEV is
                # deliberately not a fact source for this placement.
                pivot = self._provenance.pivot_fraction_chord
                lever = -pivot
                lev_q_kinematic = (
                    float(1.0 + lever * np.sin(alpha) * alpha_rate),
                    float(-heave_rate + lever * np.cos(alpha) * alpha_rate),
                )
                old_tev_q = _direct_vatistas2_velocity(
                    target=lev_edge_anchor,
                    source_x=old_tev_x,
                    source_y=old_tev_y,
                    source_gamma=old_tev_gamma,
                    core_radius=self._provenance.resolved_core_radius_chord,
                )
                old_lev_q = _direct_vatistas2_velocity(
                    target=lev_edge_anchor,
                    source_x=old_lev_x,
                    source_y=old_lev_y,
                    source_gamma=old_lev_gamma,
                    core_radius=self._provenance.resolved_core_radius_chord,
                )
                lev_q_old_wake = (
                    float(old_tev_q[0] + old_lev_q[0]),
                    float(old_tev_q[1] + old_lev_q[1]),
                )
                tev_birth = _finite_pair("captured provisional TEV point", columns[0])
                lev_q_provisional = _direct_vatistas2_velocity(
                    target=lev_edge_anchor,
                    source_x=np.asarray([tev_birth[0]], dtype=float),
                    source_y=np.asarray([tev_birth[1]], dtype=float),
                    source_gamma=np.asarray(
                        [gamma_tev_te_only_provisional], dtype=float
                    ),
                    core_radius=self._provenance.resolved_core_radius_chord,
                )
            lev_placement = _source_placement(
                family="lev",
                mode=lev_mode,
                edge_anchor=lev_edge_anchor,
                birth_position=(columns[1] if active else None),
                delta_time_convective=dt_step,
                q_kinematic=lev_q_kinematic,
                q_old_wake=lev_q_old_wake,
                q_provisional_tev=lev_q_provisional,
                continuous_parent_source_id=lev_parent_id,
                continuous_parent_position=(
                    lev_continuous_parent if lev_mode == "continuous" else None
                ),
            )
            event = DVMSourceEvent(
                enabled=True,
                status=SOURCE_STATUS,
                delta_time_convective=dt_step,
                a0_pre=a0_pre,
                a0_post=a0_post,
                lesp_critical=float(self._threshold.value),
                lesp_active=active,
                lesp_signed_target=signed_target,
                lesp_constraint_residual=(0.0 if residual == 0.0 else residual),
                lev_birth_mode=birth_mode,
                restart=restart,
                gamma_lev_new_over_u_c=gamma_lev,
                gamma_tev_new_solved_over_u_c=gamma_tev_solved,
                gamma_tev_new_persisted_over_u_c=gamma_tev_persisted,
                lev_placement=lev_placement,
                tev_placement=tev_placement,
                lev_birth_position_over_chord_backend_world=(
                    lev_placement.birth_position_over_chord_backend_world
                ),
                tev_birth_position_over_chord_backend_world=(
                    tev_placement.birth_position_over_chord_backend_world
                ),
                kelvin_residual_over_u_c=ledger.kelvin_solve_residual,
                kelvin_ledger=ledger,
                lineage=self._lineage(
                    source_step_index=source_step,
                    parent_step_index=parent_step,
                    tev_before=tev_before,
                    tev_after=tev_after,
                    lev_before=lev_before,
                    lev_after=lev_after,
                    lev_active=active,
                    first_tev_zeroed=first_tev_zeroed,
                ),
                provenance=self._provenance,
                parent_event_manifest_sha256=(self._last_enabled_event_manifest_sha256),
                producer_manifest_sha256="",
            )
            attested = _attest_direct_event(event)
            self._last_enabled_event_manifest_sha256 = attested.producer_manifest_sha256
            return attested
        except Exception:
            # A backend failure may leave partial mutations.  Refuse every later
            # enabled call rather than pretending that the history is intact.
            self._poisoned = True
            raise


__all__ = [
    "ALLOWED_THRESHOLD_SOURCE_ROLES",
    "CANONICAL_BLOCKER",
    "DVMKelvinLedger",
    "DVMSourceEvent",
    "DVMSourceLineage",
    "DVMSourcePlacement",
    "DVMSourceProvenance",
    "SOURCE_BACKEND_ID",
    "SOURCE_INTERFACE_ID",
    "SOURCE_PLACEMENT_SCHEMA_ID",
    "SOURCE_STATUS",
    "EVENT_CHAIN_DOMAIN",
    "V5hDVMSource",
    "validate_dvm_source_event",
]
