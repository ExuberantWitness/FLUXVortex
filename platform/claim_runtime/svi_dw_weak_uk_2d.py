"""Provisional formula-audit ledgers for the N2.6e1bc weak-UK shadow.

This module implements no aerodynamic solver and selects no wake state.  It
only evaluates four typed algebraic ledgers from already supplied stage data:

* material-wake advection and potential-jump conservation;
* the Kelvin bound-plus-material-wake circulation inventory;
* a manufactured weak trailing-edge unsteady-Kutta balance; and
* the trailing-edge control-volume transverse-momentum kill guard.

The weak-UK pressure jump must be explicitly tagged as coming from unsteady
Bernoulli at the same stage.  There is no endpoint sheet strength, steady
pressure-root, target load, least-squares closure, core, clamp, or damping in
this module.

The physical moving-control-volume reduction is still under independent
derivation.  Passing these algebraic ledgers is therefore not an A0 physics
GO and does not authorize a solver or production integration.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .svi_dw_types import SVIDWValidationError


_VALID_PRESSURE_PROVENANCE = {
    "unsteady_bernoulli",
    "manufactured",
}
_VALID_VORTICITY_FLUX_PROVENANCE = {
    "control_volume_quadrature",
    "manufactured",
}


def _finite_scalar(name: str, value: Any) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise SVIDWValidationError(f"{name} must be finite")
    return result


def _positive_scalar(name: str, value: Any) -> float:
    result = _finite_scalar(name, value)
    if result <= 0.0:
        raise SVIDWValidationError(f"{name} must be positive")
    return result


def _nonnegative_scalar(name: str, value: Any) -> float:
    result = _finite_scalar(name, value)
    if result < 0.0:
        raise SVIDWValidationError(f"{name} must be non-negative")
    return result


def _identifier(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise SVIDWValidationError(f"{name} must be a non-empty string")
    return value


def _readonly_array(
    name: str,
    value: Any,
    *,
    shape: tuple[int, ...] | None = None,
    ndim: int | None = None,
) -> np.ndarray:
    result = np.array(value, dtype=float, copy=True)
    if shape is not None and result.shape != shape:
        raise SVIDWValidationError(
            f"{name} must have shape {shape}, got {result.shape}"
        )
    if ndim is not None and result.ndim != ndim:
        raise SVIDWValidationError(
            f"{name} must have ndim={ndim}, got {result.shape}"
        )
    if not np.all(np.isfinite(result)):
        raise SVIDWValidationError(f"{name} contains non-finite values")
    result.setflags(write=False)
    return result


def _readonly_indices(
    name: str,
    value: Any,
    *,
    shape_tail: tuple[int, ...],
) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1 + len(shape_tail) or raw.shape[1:] != shape_tail:
        raise SVIDWValidationError(
            f"{name} must have shape (n,{','.join(map(str, shape_tail))})"
        )
    if not np.issubdtype(raw.dtype, np.integer):
        raise SVIDWValidationError(f"{name} must contain integer indices")
    result = np.array(raw, dtype=np.int64, copy=True)
    result.setflags(write=False)
    return result


def _maximum_absolute(value: Any) -> float:
    array = np.asarray(value, dtype=float)
    return float(array.max(initial=0.0)) if array.size else 0.0


def _maximum_row_norm(value: Any) -> float:
    array = np.asarray(value, dtype=float)
    if len(array) == 0:
        return 0.0
    return float(np.linalg.norm(array, axis=1).max())


def _same_density(first: float, second: float) -> bool:
    scale = max(abs(first), abs(second), 1.0)
    return abs(first - second) <= 64.0 * np.finfo(float).eps * scale


@dataclass(frozen=True)
class WeakUKReferenceScales2D:
    """Positive dimensional scales for all A0 normalized residuals.

    ``chord`` has length units, ``velocity`` has length/time units, and
    ``density`` has mass/volume units.  The resulting ledger scales are
    ``U*c`` for circulation, ``U**2`` for circulation rate, and
    ``rho*U**2*c`` for two-dimensional force per span.
    """

    chord: float
    velocity: float
    density: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "chord", _positive_scalar("chord", self.chord)
        )
        object.__setattr__(
            self,
            "velocity",
            _positive_scalar("velocity", self.velocity),
        )
        object.__setattr__(
            self,
            "density",
            _positive_scalar("density", self.density),
        )

    @property
    def circulation(self) -> float:
        return self.velocity * self.chord

    @property
    def circulation_rate(self) -> float:
        return self.velocity**2

    @property
    def force_per_span(self) -> float:
        return self.density * self.velocity**2 * self.chord


@dataclass(frozen=True)
class MaterialWakeState2D:
    """One explicitly present material-wake inventory at a named stage.

    ``potential_jump`` is a nodal material scalar.  ``oriented_edges`` stores
    explicit upstream-to-downstream node-index pairs.  Wake circulation is
    the gauge-invariant sum of ``mu[downstream]-mu[upstream]`` over those
    edges, plus an explicitly separate far point-vortex circulation.

    An explicit empty state represents the zero-history limit; ``None`` is
    never accepted as a substitute for history.
    """

    history_id: str
    stage_id: str
    time: float
    material_ids: tuple[str, ...]
    positions: Any
    potential_jump: Any
    oriented_edges: Any
    far_point_vortex_circulation: float = 0.0

    def __post_init__(self) -> None:
        history_id = _identifier("history_id", self.history_id)
        stage_id = _identifier("stage_id", self.stage_id)
        time = _finite_scalar("time", self.time)
        material_ids = tuple(self.material_ids)
        if any(
            not isinstance(item, str) or not item
            for item in material_ids
        ):
            raise SVIDWValidationError(
                "material_ids must contain non-empty strings"
            )
        if len(set(material_ids)) != len(material_ids):
            raise SVIDWValidationError("material_ids must be unique")
        count = len(material_ids)
        positions = _readonly_array(
            "positions", self.positions, shape=(count, 2)
        )
        jump = _readonly_array(
            "potential_jump",
            self.potential_jump,
            shape=(count,),
        )
        edges = _readonly_indices(
            "oriented_edges",
            self.oriented_edges,
            shape_tail=(2,),
        )
        if (
            np.any(edges < 0)
            or np.any(edges >= count)
            or np.any(edges[:, 0] == edges[:, 1])
        ):
            raise SVIDWValidationError(
                "oriented_edges must connect distinct valid material nodes"
            )
        if len(edges) != len({tuple(edge) for edge in edges.tolist()}):
            raise SVIDWValidationError(
                "oriented_edges must not contain duplicate directed edges"
            )
        far_circulation = _finite_scalar(
            "far_point_vortex_circulation",
            self.far_point_vortex_circulation,
        )
        object.__setattr__(self, "history_id", history_id)
        object.__setattr__(self, "stage_id", stage_id)
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "material_ids", material_ids)
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "potential_jump", jump)
        object.__setattr__(self, "oriented_edges", edges)
        object.__setattr__(
            self,
            "far_point_vortex_circulation",
            far_circulation,
        )

    @classmethod
    def empty(
        cls,
        *,
        history_id: str,
        stage_id: str,
        time: float,
    ) -> "MaterialWakeState2D":
        """Return an explicit typed zero-history state."""

        return cls(
            history_id=history_id,
            stage_id=stage_id,
            time=time,
            material_ids=(),
            positions=np.empty((0, 2), dtype=float),
            potential_jump=np.empty(0, dtype=float),
            oriented_edges=np.empty((0, 2), dtype=np.int64),
            far_point_vortex_circulation=0.0,
        )

    @property
    def edge_circulation(self) -> np.ndarray:
        result = (
            self.potential_jump[self.oriented_edges[:, 1]]
            - self.potential_jump[self.oriented_edges[:, 0]]
        )
        result.setflags(write=False)
        return result

    @property
    def total_edge_circulation(self) -> float:
        return float(np.sum(self.edge_circulation))

    @property
    def total_circulation(self) -> float:
        return (
            self.total_edge_circulation
            + self.far_point_vortex_circulation
        )


@dataclass(frozen=True)
class MaterialWakeTransportLedger2D:
    retained_material_ids: tuple[str, ...]
    newborn_material_ids: tuple[str, ...]
    position_rate: np.ndarray
    position_residual: np.ndarray
    potential_jump_rate: np.ndarray
    maximum_position_scaled_residual: float
    maximum_jump_scaled_residual: float
    maximum_jump_mutation: float
    passed: bool


def material_wake_transport_ledger(
    previous: MaterialWakeState2D,
    current: MaterialWakeState2D,
    *,
    stage_velocity: Any,
    scales: WeakUKReferenceScales2D,
    tolerance: float = 1.0e-12,
) -> MaterialWakeTransportLedger2D:
    """Check ``X_dot=u_bar`` and ``D mu/Dt=0`` without updating history."""

    if not isinstance(previous, MaterialWakeState2D):
        raise SVIDWValidationError(
            "previous must be MaterialWakeState2D; missing history is invalid"
        )
    if not isinstance(current, MaterialWakeState2D):
        raise SVIDWValidationError(
            "current must be MaterialWakeState2D; missing history is invalid"
        )
    if not isinstance(scales, WeakUKReferenceScales2D):
        raise SVIDWValidationError(
            "scales must be WeakUKReferenceScales2D"
        )
    tol = _nonnegative_scalar("tolerance", tolerance)
    if previous.history_id != current.history_id:
        raise SVIDWValidationError(
            "material wake history_id changed across the step"
        )
    current_index = {
        material_id: index
        for index, material_id in enumerate(current.material_ids)
    }
    missing_old = tuple(
        material_id
        for material_id in previous.material_ids
        if material_id not in current_index
    )
    if missing_old:
        raise SVIDWValidationError(
            f"retained material wake nodes disappeared: {missing_old}"
        )
    retained_ids = previous.material_ids
    newborn_ids = tuple(
        material_id
        for material_id in current.material_ids
        if material_id not in set(previous.material_ids)
    )
    retained_current_indices = np.array(
        [current_index[item] for item in retained_ids],
        dtype=np.int64,
    )
    timestep = current.time - previous.time
    if not np.isfinite(timestep) or timestep <= 0.0:
        raise SVIDWValidationError(
            "current material-wake time must exceed previous time"
        )
    count = len(retained_ids)
    velocity = _readonly_array(
        "stage_velocity", stage_velocity, shape=(count, 2)
    )
    current_old_positions = current.positions[retained_current_indices]
    current_old_jump = current.potential_jump[retained_current_indices]
    position_rate = (
        current_old_positions - previous.positions
    ) / timestep
    position_residual = position_rate - velocity
    jump_mutation = current_old_jump - previous.potential_jump
    jump_rate = jump_mutation / timestep
    position_scaled = (
        _maximum_row_norm(position_residual) / scales.velocity
    )
    jump_scaled = (
        _maximum_absolute(np.abs(jump_rate))
        / scales.circulation_rate
    )
    maximum_mutation = _maximum_absolute(np.abs(jump_mutation))
    arrays = []
    for value in (position_rate, position_residual, jump_rate):
        array = np.array(value, dtype=float, copy=True)
        array.setflags(write=False)
        arrays.append(array)
    return MaterialWakeTransportLedger2D(
        retained_material_ids=retained_ids,
        newborn_material_ids=newborn_ids,
        position_rate=arrays[0],
        position_residual=arrays[1],
        potential_jump_rate=arrays[2],
        maximum_position_scaled_residual=position_scaled,
        maximum_jump_scaled_residual=jump_scaled,
        maximum_jump_mutation=maximum_mutation,
        passed=max(position_scaled, jump_scaled) <= tol,
    )


@dataclass(frozen=True)
class KelvinMaterialWakeLedger2D:
    stage_id: str
    bound_circulation: float
    wake_edge_circulation: float
    far_point_vortex_circulation: float
    wake_circulation: float
    reference_circulation: float
    residual: float
    scaled_residual: float
    passed: bool


def kelvin_material_wake_ledger(
    *,
    bound_circulation: float,
    wake: MaterialWakeState2D,
    reference_circulation: float,
    scales: WeakUKReferenceScales2D,
    tolerance: float = 1.0e-12,
) -> KelvinMaterialWakeLedger2D:
    """Check bound plus gauge-invariant oriented-edge wake circulation."""

    if not isinstance(wake, MaterialWakeState2D):
        raise SVIDWValidationError(
            "wake must be MaterialWakeState2D; missing history is invalid"
        )
    if not isinstance(scales, WeakUKReferenceScales2D):
        raise SVIDWValidationError(
            "scales must be WeakUKReferenceScales2D"
        )
    bound = _finite_scalar("bound_circulation", bound_circulation)
    reference = _finite_scalar(
        "reference_circulation", reference_circulation
    )
    tol = _nonnegative_scalar("tolerance", tolerance)
    wake_circulation = wake.total_circulation
    residual = bound + wake_circulation - reference
    scaled = abs(residual) / scales.circulation
    return KelvinMaterialWakeLedger2D(
        stage_id=wake.stage_id,
        bound_circulation=bound,
        wake_edge_circulation=wake.total_edge_circulation,
        far_point_vortex_circulation=(
            wake.far_point_vortex_circulation
        ),
        wake_circulation=wake_circulation,
        reference_circulation=reference,
        residual=float(residual),
        scaled_residual=float(scaled),
        passed=scaled <= tol,
    )


@dataclass(frozen=True)
class TEControlVolume2D:
    """One explicitly closed, oriented trailing-edge control polygon."""

    control_volume_id: str
    stage_id: str
    vertices: Any

    def __post_init__(self) -> None:
        control_volume_id = _identifier(
            "control_volume_id", self.control_volume_id
        )
        stage_id = _identifier("stage_id", self.stage_id)
        vertices = _readonly_array("vertices", self.vertices, ndim=2)
        if vertices.shape[1:] != (2,) or len(vertices) < 4:
            raise SVIDWValidationError(
                "control-volume vertices must have shape (n>=4,2)"
            )
        scale = max(
            float(np.max(np.abs(vertices), initial=0.0)),
            1.0,
        )
        closure_tolerance = 64.0 * np.finfo(float).eps * scale
        if np.linalg.norm(vertices[-1] - vertices[0]) > closure_tolerance:
            raise SVIDWValidationError(
                "trailing-edge control volume must be explicitly closed"
            )
        edges = np.diff(vertices, axis=0)
        lengths = np.linalg.norm(edges, axis=1)
        if np.any(lengths <= closure_tolerance):
            raise SVIDWValidationError(
                "control-volume boundary contains a zero-length segment"
            )
        cross_sum = np.sum(
            vertices[:-1, 0] * vertices[1:, 1]
            - vertices[1:, 0] * vertices[:-1, 1]
        )
        signed_area = 0.5 * float(cross_sum)
        if abs(signed_area) <= closure_tolerance**2:
            raise SVIDWValidationError(
                "trailing-edge control volume has zero signed area"
            )
        object.__setattr__(
            self, "control_volume_id", control_volume_id
        )
        object.__setattr__(self, "stage_id", stage_id)
        object.__setattr__(self, "vertices", vertices)

    @property
    def segment_count(self) -> int:
        return len(self.vertices) - 1

    @property
    def segment_vectors(self) -> np.ndarray:
        result = np.diff(self.vertices, axis=0)
        result.setflags(write=False)
        return result

    @property
    def segment_lengths(self) -> np.ndarray:
        result = np.linalg.norm(self.segment_vectors, axis=1)
        result.setflags(write=False)
        return result

    @property
    def signed_area(self) -> float:
        return 0.5 * float(
            np.sum(
                self.vertices[:-1, 0] * self.vertices[1:, 1]
                - self.vertices[1:, 0] * self.vertices[:-1, 1]
            )
        )

    @property
    def outward_normals(self) -> np.ndarray:
        tangent = self.segment_vectors / self.segment_lengths[:, None]
        if self.signed_area > 0.0:
            result = np.column_stack((tangent[:, 1], -tangent[:, 0]))
        else:
            result = np.column_stack((-tangent[:, 1], tangent[:, 0]))
        result.setflags(write=False)
        return result


@dataclass(frozen=True)
class TEVorticityFluxLedger2D:
    control_volume_id: str
    stage_id: str
    provenance: str
    relative_normal_velocity: np.ndarray
    segment_flux: np.ndarray
    total_flux: float
    scaled_flux: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "control_volume_id",
            _identifier("control_volume_id", self.control_volume_id),
        )
        object.__setattr__(
            self, "stage_id", _identifier("stage_id", self.stage_id)
        )
        if self.provenance not in _VALID_VORTICITY_FLUX_PROVENANCE:
            raise SVIDWValidationError(
                "vorticity-flux provenance must be "
                "control_volume_quadrature or manufactured"
            )
        relative = _readonly_array(
            "relative_normal_velocity",
            self.relative_normal_velocity,
            ndim=1,
        )
        segment = _readonly_array(
            "segment_flux",
            self.segment_flux,
            shape=relative.shape,
        )
        total = _finite_scalar("total_flux", self.total_flux)
        scaled = _finite_scalar("scaled_flux", self.scaled_flux)
        object.__setattr__(
            self, "relative_normal_velocity", relative
        )
        object.__setattr__(self, "segment_flux", segment)
        object.__setattr__(self, "total_flux", total)
        object.__setattr__(self, "scaled_flux", scaled)


def te_control_volume_vorticity_flux(
    control_volume: TEControlVolume2D,
    *,
    vorticity: Any,
    fluid_velocity: Any,
    control_volume_velocity: Any,
    provenance: str,
    scales: WeakUKReferenceScales2D,
) -> TEVorticityFluxLedger2D:
    """Evaluate ``integral omega*(u-v_CV).n ds`` on a closed polygon."""

    if not isinstance(control_volume, TEControlVolume2D):
        raise SVIDWValidationError(
            "control_volume must be TEControlVolume2D"
        )
    if not isinstance(scales, WeakUKReferenceScales2D):
        raise SVIDWValidationError(
            "scales must be WeakUKReferenceScales2D"
        )
    if provenance not in _VALID_VORTICITY_FLUX_PROVENANCE:
        raise SVIDWValidationError(
            "vorticity-flux provenance must be "
            "control_volume_quadrature or manufactured; "
            "Gamma_birth/dt is forbidden"
        )
    count = control_volume.segment_count
    omega = _readonly_array("vorticity", vorticity, shape=(count,))
    fluid = _readonly_array(
        "fluid_velocity", fluid_velocity, shape=(count, 2)
    )
    cv_velocity = _readonly_array(
        "control_volume_velocity",
        control_volume_velocity,
        shape=(count, 2),
    )
    relative_normal = np.einsum(
        "ij,ij->i",
        fluid - cv_velocity,
        control_volume.outward_normals,
    )
    segment_flux = (
        omega * relative_normal * control_volume.segment_lengths
    )
    total = float(np.sum(segment_flux))
    arrays = []
    for value in (relative_normal, segment_flux):
        array = np.array(value, dtype=float, copy=True)
        array.setflags(write=False)
        arrays.append(array)
    return TEVorticityFluxLedger2D(
        control_volume_id=control_volume.control_volume_id,
        stage_id=control_volume.stage_id,
        provenance=provenance,
        relative_normal_velocity=arrays[0],
        segment_flux=arrays[1],
        total_flux=total,
        scaled_flux=total / scales.circulation_rate,
    )


@dataclass(frozen=True)
class BoundCirculationStep2D:
    """Two real bound-circulation levels defining one stage rate."""

    stage_id: str
    previous: float
    current: float
    timestep: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "stage_id", _identifier("stage_id", self.stage_id)
        )
        object.__setattr__(
            self, "previous", _finite_scalar("previous", self.previous)
        )
        object.__setattr__(
            self, "current", _finite_scalar("current", self.current)
        )
        object.__setattr__(
            self,
            "timestep",
            _positive_scalar("timestep", self.timestep),
        )

    @property
    def rate(self) -> float:
        return (self.current - self.previous) / self.timestep


@dataclass(frozen=True)
class TEBernoulliPressureJump2D:
    """Same-stage TE pressure limits from the unsteady Bernoulli observer."""

    stage_id: str
    pressure_lower: float
    pressure_upper: float
    density: float
    provenance: str = "unsteady_bernoulli"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "stage_id", _identifier("stage_id", self.stage_id)
        )
        object.__setattr__(
            self,
            "pressure_lower",
            _finite_scalar("pressure_lower", self.pressure_lower),
        )
        object.__setattr__(
            self,
            "pressure_upper",
            _finite_scalar("pressure_upper", self.pressure_upper),
        )
        object.__setattr__(
            self,
            "density",
            _positive_scalar("density", self.density),
        )
        if self.provenance not in _VALID_PRESSURE_PROVENANCE:
            raise SVIDWValidationError(
                "TE pressure provenance must be unsteady_bernoulli or "
                "explicit manufactured data; weak_uk_backsolve is forbidden"
            )

    @property
    def lower_minus_upper(self) -> float:
        return self.pressure_lower - self.pressure_upper


@dataclass(frozen=True)
class WeakUnsteadyKuttaLedger2D:
    stage_id: str
    bound_circulation_rate: float
    control_volume_vorticity_flux: float
    specific_pressure_jump: float
    residual: float
    scaled_residual: float
    passed: bool


def weak_unsteady_kutta_ledger(
    circulation: BoundCirculationStep2D,
    vorticity_flux: TEVorticityFluxLedger2D,
    pressure_jump: TEBernoulliPressureJump2D,
    *,
    scales: WeakUKReferenceScales2D,
    tolerance: float = 1.0e-12,
) -> WeakUnsteadyKuttaLedger2D:
    """Check weak UK without solving for a root or endpoint strength."""

    if not isinstance(circulation, BoundCirculationStep2D):
        raise SVIDWValidationError(
            "circulation must be BoundCirculationStep2D"
        )
    if not isinstance(vorticity_flux, TEVorticityFluxLedger2D):
        raise SVIDWValidationError(
            "vorticity_flux must be TEVorticityFluxLedger2D"
        )
    if not isinstance(pressure_jump, TEBernoulliPressureJump2D):
        raise SVIDWValidationError(
            "pressure_jump must be TEBernoulliPressureJump2D"
        )
    if not isinstance(scales, WeakUKReferenceScales2D):
        raise SVIDWValidationError(
            "scales must be WeakUKReferenceScales2D"
        )
    tol = _nonnegative_scalar("tolerance", tolerance)
    stage_ids = {
        circulation.stage_id,
        vorticity_flux.stage_id,
        pressure_jump.stage_id,
    }
    if len(stage_ids) != 1:
        raise SVIDWValidationError(
            "weak-UK inputs must belong to the same converged stage"
        )
    if not _same_density(pressure_jump.density, scales.density):
        raise SVIDWValidationError(
            "pressure-jump density does not match the reference density"
        )
    specific_pressure_jump = (
        pressure_jump.lower_minus_upper / pressure_jump.density
    )
    residual = (
        circulation.rate
        + vorticity_flux.total_flux
        - specific_pressure_jump
    )
    scaled = abs(residual) / scales.circulation_rate
    return WeakUnsteadyKuttaLedger2D(
        stage_id=circulation.stage_id,
        bound_circulation_rate=float(circulation.rate),
        control_volume_vorticity_flux=(
            vorticity_flux.total_flux
        ),
        specific_pressure_jump=float(specific_pressure_jump),
        residual=float(residual),
        scaled_residual=float(scaled),
        passed=scaled <= tol,
    )


@dataclass(frozen=True)
class TETransverseMomentumLedger2D:
    control_volume_id: str
    stage_id: str
    forming_angle_rad: float
    transverse_direction: np.ndarray
    storage_rate: np.ndarray
    convective_flux: np.ndarray
    pressure_flux: np.ndarray
    viscous_flux: np.ndarray
    vector_residual: np.ndarray
    transverse_residual: float
    scaled_transverse_residual: float
    passed: bool


def te_transverse_momentum_ledger(
    control_volume: TEControlVolume2D,
    *,
    density: float,
    previous_area_momentum: Any,
    current_area_momentum: Any,
    timestep: float,
    fluid_velocity: Any,
    control_volume_velocity: Any,
    pressure: Any,
    viscous_traction: Any,
    forming_angle_rad: float,
    scales: WeakUKReferenceScales2D,
    tolerance: float = 1.0e-12,
) -> TETransverseMomentumLedger2D:
    """Evaluate the independent TE transverse-momentum kill guard.

    ``previous_area_momentum`` and ``current_area_momentum`` are the
    area-integrated vectors ``integral_A rho*u dA``.  ``viscous_traction`` is
    the boundary vector ``tau.n``; its flux enters with the frozen minus sign.
    The returned transverse residual is an observation only and is never used
    to select a forming angle in this module.
    """

    if not isinstance(control_volume, TEControlVolume2D):
        raise SVIDWValidationError(
            "control_volume must be TEControlVolume2D"
        )
    if not isinstance(scales, WeakUKReferenceScales2D):
        raise SVIDWValidationError(
            "scales must be WeakUKReferenceScales2D"
        )
    rho = _positive_scalar("density", density)
    if not _same_density(rho, scales.density):
        raise SVIDWValidationError(
            "momentum-ledger density does not match reference density"
        )
    dt = _positive_scalar("timestep", timestep)
    angle = _finite_scalar("forming_angle_rad", forming_angle_rad)
    tol = _nonnegative_scalar("tolerance", tolerance)
    previous = _readonly_array(
        "previous_area_momentum",
        previous_area_momentum,
        shape=(2,),
    )
    current = _readonly_array(
        "current_area_momentum",
        current_area_momentum,
        shape=(2,),
    )
    count = control_volume.segment_count
    fluid = _readonly_array(
        "fluid_velocity", fluid_velocity, shape=(count, 2)
    )
    cv_velocity = _readonly_array(
        "control_volume_velocity",
        control_volume_velocity,
        shape=(count, 2),
    )
    pressure_field = _readonly_array(
        "pressure", pressure, shape=(count,)
    )
    traction = _readonly_array(
        "viscous_traction", viscous_traction, shape=(count, 2)
    )
    normals = control_volume.outward_normals
    lengths = control_volume.segment_lengths
    relative_normal = np.einsum(
        "ij,ij->i", fluid - cv_velocity, normals
    )
    storage = (current - previous) / dt
    convective = np.sum(
        rho
        * fluid
        * relative_normal[:, None]
        * lengths[:, None],
        axis=0,
    )
    pressure_flux = np.sum(
        pressure_field[:, None] * normals * lengths[:, None],
        axis=0,
    )
    viscous_flux = np.sum(traction * lengths[:, None], axis=0)
    vector_residual = (
        storage + convective + pressure_flux - viscous_flux
    )
    transverse = np.array(
        (-np.sin(angle), np.cos(angle)), dtype=float
    )
    transverse_residual = float(np.dot(transverse, vector_residual))
    scaled = abs(transverse_residual) / scales.force_per_span
    arrays = []
    for value in (
        transverse,
        storage,
        convective,
        pressure_flux,
        viscous_flux,
        vector_residual,
    ):
        array = np.array(value, dtype=float, copy=True)
        array.setflags(write=False)
        arrays.append(array)
    return TETransverseMomentumLedger2D(
        control_volume_id=control_volume.control_volume_id,
        stage_id=control_volume.stage_id,
        forming_angle_rad=angle,
        transverse_direction=arrays[0],
        storage_rate=arrays[1],
        convective_flux=arrays[2],
        pressure_flux=arrays[3],
        viscous_flux=arrays[4],
        vector_residual=arrays[5],
        transverse_residual=transverse_residual,
        scaled_transverse_residual=float(scaled),
        passed=scaled <= tol,
    )
