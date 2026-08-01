"""Independent near-wall field data contract for claim N2.6b4f.

This module does not close a boundary layer and does not predict aerodynamic
loads.  It makes the evidence required by a future profile/edge closure
machine-checkable:

* time-resolved fields sampled along named wall-normal rays;
* moving-wall and edge kinematics in one objective frame;
* surface topology and side identity needed for spatial derivatives; and
* provenance/coverage that cannot be silently replaced by integrated loads.

Manufactured data can validate the schema and geometric identities, but are
explicitly ineligible for physical model promotion.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np


class NearWallFieldContractError(ValueError):
    """Malformed field evidence or a forbidden target channel."""


_FORBIDDEN_KEY_FRAGMENTS = (
    "lift",
    "thrust",
    "force",
    "moment_target",
    "lesp",
    "pressure_target",
    "pressure_residual",
    "structural",
)

_ALLOWED_SOURCE_KINDS = frozenset({
    "manufactured",
    "differential_boundary_layer",
    "wall_resolved_rans",
    "les",
    "dns",
    "experiment",
})

_PHYSICAL_SOURCE_KINDS = _ALLOWED_SOURCE_KINDS-frozenset({"manufactured"})


def _normalise_key(key: object) -> str:
    return str(key).strip().lower().replace("-", "_").replace(" ", "_")


def assert_no_forbidden_targets(
    value: object,
    *,
    path: str = "dataset",
) -> None:
    """Reject load/pressure-residual/structure labels anywhere in a mapping.

    The check intentionally operates before a typed dataset is constructed so
    that extra target arrays cannot be hidden in an otherwise valid archive.
    """
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalised = _normalise_key(key)
            if any(fragment in normalised for fragment in _FORBIDDEN_KEY_FRAGMENTS):
                raise NearWallFieldContractError(
                    f"forbidden target channel at {path}.{key}"
                )
            assert_no_forbidden_targets(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            assert_no_forbidden_targets(child, path=f"{path}[{index}]")


def _array(name: str, value, *, ndim: int | None = None) -> np.ndarray:
    result = np.asarray(value)
    if ndim is not None and result.ndim != ndim:
        raise NearWallFieldContractError(
            f"{name} must have {ndim} dimensions, got {result.ndim}"
        )
    if not np.all(np.isfinite(result)):
        raise NearWallFieldContractError(f"{name} contains non-finite values")
    return result.copy()


@dataclass(frozen=True)
class NearWallFieldProvenance:
    """Auditable origin and regime coverage of one field data set."""

    source_kind: str
    reference: str
    case_id: str
    split_role: str
    independent_audit_id: str
    edge_convention: str
    reynolds_min: float
    reynolds_max: float
    regime_flags: frozenset[str]
    independent_of_load_targets: bool = True

    def __post_init__(self) -> None:
        source_kind = str(self.source_kind).strip().lower()
        if source_kind not in _ALLOWED_SOURCE_KINDS:
            raise NearWallFieldContractError(
                f"unknown source_kind {self.source_kind!r}"
            )
        split_role = str(self.split_role).strip().lower()
        if split_role not in {"train", "validation", "test"}:
            raise NearWallFieldContractError(
                "split_role must be train, validation or test"
            )
        for name in ("reference", "case_id", "edge_convention"):
            if not str(getattr(self, name)).strip():
                raise NearWallFieldContractError(f"{name} must be non-empty")

        reynolds_min = float(self.reynolds_min)
        reynolds_max = float(self.reynolds_max)
        if (
            not np.isfinite(reynolds_min)
            or not np.isfinite(reynolds_max)
            or reynolds_min <= 0.0
            or reynolds_max < reynolds_min
        ):
            raise NearWallFieldContractError(
                "Reynolds interval must be finite, positive and ordered"
            )

        regime_flags = frozenset(
            str(flag).strip().lower()
            for flag in self.regime_flags
            if str(flag).strip()
        )
        object.__setattr__(self, "source_kind", source_kind)
        object.__setattr__(self, "split_role", split_role)
        object.__setattr__(self, "reference", str(self.reference).strip())
        object.__setattr__(self, "case_id", str(self.case_id).strip())
        object.__setattr__(
            self,
            "independent_audit_id",
            str(self.independent_audit_id).strip(),
        )
        object.__setattr__(
            self,
            "edge_convention",
            str(self.edge_convention).strip(),
        )
        object.__setattr__(self, "reynolds_min", reynolds_min)
        object.__setattr__(self, "reynolds_max", reynolds_max)
        object.__setattr__(self, "regime_flags", regime_flags)


@dataclass(frozen=True)
class RequiredNearWallCoverage:
    """Target-domain evidence requirements, separate from field values."""

    reynolds_min: float
    reynolds_max: float
    required_regimes: frozenset[str]
    required_split_role: str = "test"

    def __post_init__(self) -> None:
        reynolds_min = float(self.reynolds_min)
        reynolds_max = float(self.reynolds_max)
        if reynolds_min <= 0.0 or reynolds_max < reynolds_min:
            raise NearWallFieldContractError("invalid target Reynolds interval")
        split = str(self.required_split_role).strip().lower()
        if split not in {"train", "validation", "test"}:
            raise NearWallFieldContractError("invalid required_split_role")
        regimes = frozenset(
            str(flag).strip().lower()
            for flag in self.required_regimes
            if str(flag).strip()
        )
        object.__setattr__(self, "reynolds_min", reynolds_min)
        object.__setattr__(self, "reynolds_max", reynolds_max)
        object.__setattr__(self, "required_split_role", split)
        object.__setattr__(self, "required_regimes", regimes)


@dataclass(frozen=True)
class NearWallFieldDataset:
    """Time-resolved field samples on a connected moving surface."""

    provenance: NearWallFieldProvenance
    time: np.ndarray
    surface_chart: np.ndarray
    surface_faces: np.ndarray
    side: np.ndarray
    normal_coordinate: np.ndarray
    density: np.ndarray
    velocity: np.ndarray
    wall_position: np.ndarray
    wall_velocity: np.ndarray
    surface_normal: np.ndarray
    tangent_basis: np.ndarray
    edge_position: np.ndarray
    edge_velocity: np.ndarray
    extraction_tolerance: float

    def __post_init__(self) -> None:
        if not isinstance(self.provenance, NearWallFieldProvenance):
            raise NearWallFieldContractError(
                "provenance must be NearWallFieldProvenance"
            )
        for name in (
            "time",
            "surface_chart",
            "normal_coordinate",
            "density",
            "velocity",
            "wall_position",
            "wall_velocity",
            "surface_normal",
            "tangent_basis",
            "edge_position",
            "edge_velocity",
        ):
            object.__setattr__(
                self,
                name,
                _array(name, getattr(self, name)),
            )

        faces = np.asarray(self.surface_faces)
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise NearWallFieldContractError(
                "surface_faces must have shape (nf,3)"
            )
        if not np.issubdtype(faces.dtype, np.integer):
            if not np.all(np.equal(faces, np.floor(faces))):
                raise NearWallFieldContractError(
                    "surface_faces must contain integer indices"
                )
        object.__setattr__(self, "surface_faces", faces.astype(int, copy=True))

        side = np.asarray(self.side)
        if side.ndim != 1 or not np.all(np.isin(side, (-1, 1))):
            raise NearWallFieldContractError(
                "side must be one-dimensional with values -1 or +1"
            )
        object.__setattr__(self, "side", side.astype(int, copy=True))

        tolerance = float(self.extraction_tolerance)
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise NearWallFieldContractError(
                "extraction_tolerance must be finite and positive"
            )
        object.__setattr__(self, "extraction_tolerance", tolerance)


@dataclass(frozen=True)
class NearWallFieldContractReport:
    """Schema/identity residuals and an explicit evidence-coverage verdict."""

    schema_valid: bool
    identity_valid: bool
    coverage_complete: bool
    production_evidence_eligible: bool
    source_kind: str
    case_id: str
    edge_convention: str
    max_no_slip_error: float
    max_edge_velocity_error: float
    max_edge_position_error: float
    max_normal_norm_error: float
    max_basis_norm_error: float
    max_basis_orthogonality_error: float
    max_handedness_error: float
    missing_coverage: tuple[str, ...]


def _max_norm(value: np.ndarray) -> float:
    return float(np.max(np.linalg.norm(value, axis=-1)))


def _validate_shapes(dataset: NearWallFieldDataset) -> tuple[int, int, int]:
    time = dataset.time
    if time.ndim != 1 or len(time) < 2 or np.any(np.diff(time) <= 0.0):
        raise NearWallFieldContractError(
            "time must be one-dimensional and strictly increasing"
        )
    nt = len(time)

    chart = dataset.surface_chart
    if chart.ndim != 2 or chart.shape[1] != 2 or len(chart) < 3:
        raise NearWallFieldContractError(
            "surface_chart must have shape (ns,2) with ns>=3"
        )
    ns = len(chart)
    if dataset.side.shape != (ns,):
        raise NearWallFieldContractError(f"side must have shape ({ns},)")
    if (
        dataset.surface_faces.size == 0
        or np.min(dataset.surface_faces) < 0
        or np.max(dataset.surface_faces) >= ns
    ):
        raise NearWallFieldContractError(
            "surface_faces must be non-empty and reference valid nodes"
        )

    coordinate = dataset.normal_coordinate
    if coordinate.ndim != 3 or coordinate.shape[:2] != (nt, ns):
        raise NearWallFieldContractError(
            "normal_coordinate must have shape (nt,ns,nn)"
        )
    nn = coordinate.shape[2]
    if nn < 2:
        raise NearWallFieldContractError("normal profiles require nn>=2")

    expected = {
        "density": (nt, ns, nn),
        "velocity": (nt, ns, nn, 3),
        "wall_position": (nt, ns, 3),
        "wall_velocity": (nt, ns, 3),
        "surface_normal": (nt, ns, 3),
        "tangent_basis": (nt, ns, 2, 3),
        "edge_position": (nt, ns, 3),
        "edge_velocity": (nt, ns, 3),
    }
    for name, shape in expected.items():
        if getattr(dataset, name).shape != shape:
            raise NearWallFieldContractError(
                f"{name} must have shape {shape}, got "
                f"{getattr(dataset, name).shape}"
            )
    if np.any(dataset.density <= 0.0):
        raise NearWallFieldContractError("density must be strictly positive")
    if np.any(np.diff(coordinate, axis=-1) <= 0.0):
        raise NearWallFieldContractError(
            "normal_coordinate must increase along every profile"
        )
    if np.max(np.abs(coordinate[..., 0])) > dataset.extraction_tolerance:
        raise NearWallFieldContractError(
            "normal_coordinate must start at the wall (zero)"
        )
    return nt, ns, nn


def _missing_coverage(
    provenance: NearWallFieldProvenance,
    requirements: RequiredNearWallCoverage | None,
) -> tuple[str, ...]:
    missing: list[str] = []
    if provenance.source_kind not in _PHYSICAL_SOURCE_KINDS:
        missing.append("independent_non_manufactured_source")
    if not provenance.independent_of_load_targets:
        missing.append("independent_of_load_targets")
    if not provenance.independent_audit_id:
        missing.append("independent_audit_id")
    if requirements is None:
        missing.append("target_coverage_requirements")
        return tuple(missing)
    if provenance.split_role != requirements.required_split_role:
        missing.append(f"split_role:{requirements.required_split_role}")
    if provenance.reynolds_min > requirements.reynolds_min:
        missing.append("reynolds_lower_bound")
    if provenance.reynolds_max < requirements.reynolds_max:
        missing.append("reynolds_upper_bound")
    for flag in sorted(requirements.required_regimes-provenance.regime_flags):
        missing.append(f"regime:{flag}")
    return tuple(missing)


def validate_near_wall_field_dataset(
    dataset: NearWallFieldDataset,
    *,
    requirements: RequiredNearWallCoverage | None = None,
) -> NearWallFieldContractReport:
    """Validate field identities without judging profile-model accuracy."""
    if not isinstance(dataset, NearWallFieldDataset):
        raise NearWallFieldContractError(
            "dataset must be NearWallFieldDataset"
        )
    _validate_shapes(dataset)

    normal = dataset.surface_normal
    tangent_1 = dataset.tangent_basis[..., 0, :]
    tangent_2 = dataset.tangent_basis[..., 1, :]
    tolerance = dataset.extraction_tolerance

    max_no_slip_error = _max_norm(
        dataset.velocity[..., 0, :]-dataset.wall_velocity
    )
    max_edge_velocity_error = _max_norm(
        dataset.velocity[..., -1, :]-dataset.edge_velocity
    )
    expected_edge = (
        dataset.wall_position
        +dataset.normal_coordinate[..., -1, None]*normal
    )
    max_edge_position_error = _max_norm(
        dataset.edge_position-expected_edge
    )
    max_normal_norm_error = float(
        np.max(np.abs(np.linalg.norm(normal, axis=-1)-1.0))
    )
    max_basis_norm_error = float(max(
        np.max(np.abs(np.linalg.norm(tangent_1, axis=-1)-1.0)),
        np.max(np.abs(np.linalg.norm(tangent_2, axis=-1)-1.0)),
    ))
    max_basis_orthogonality_error = float(max(
        np.max(np.abs(np.sum(tangent_1*tangent_2, axis=-1))),
        np.max(np.abs(np.sum(tangent_1*normal, axis=-1))),
        np.max(np.abs(np.sum(tangent_2*normal, axis=-1))),
    ))
    max_handedness_error = _max_norm(
        np.cross(tangent_1, tangent_2)-normal
    )

    identity_valid = all(
        residual <= tolerance
        for residual in (
            max_no_slip_error,
            max_edge_velocity_error,
            max_edge_position_error,
            max_normal_norm_error,
            max_basis_norm_error,
            max_basis_orthogonality_error,
            max_handedness_error,
        )
    )
    missing = _missing_coverage(dataset.provenance, requirements)
    coverage_complete = not missing
    return NearWallFieldContractReport(
        schema_valid=True,
        identity_valid=identity_valid,
        coverage_complete=coverage_complete,
        production_evidence_eligible=identity_valid and coverage_complete,
        source_kind=dataset.provenance.source_kind,
        case_id=dataset.provenance.case_id,
        edge_convention=dataset.provenance.edge_convention,
        max_no_slip_error=max_no_slip_error,
        max_edge_velocity_error=max_edge_velocity_error,
        max_edge_position_error=max_edge_position_error,
        max_normal_norm_error=max_normal_norm_error,
        max_basis_norm_error=max_basis_norm_error,
        max_basis_orthogonality_error=max_basis_orthogonality_error,
        max_handedness_error=max_handedness_error,
        missing_coverage=missing,
    )


def aggregate_missing_coverage(
    reports: Iterable[NearWallFieldContractReport],
) -> tuple[str, ...]:
    """Return coverage gaps common to an evidence suite.

    A gap is retained only when no supplied report covers it.  This helper is
    deliberately conservative: it does not turn a set of metadata reports into
    model-accuracy evidence.
    """
    reports = tuple(reports)
    if not reports:
        return ("no_evidence_reports",)
    all_gaps = set().union(*(report.missing_coverage for report in reports))
    return tuple(sorted(
        gap
        for gap in all_gaps
        if all(gap in report.missing_coverage for report in reports)
    ))

