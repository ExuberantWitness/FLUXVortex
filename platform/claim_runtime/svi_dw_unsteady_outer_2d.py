"""N2.6e1 S1 unsteady attached outer flow with one trailing-edge wake.

This module is a deliberately narrow equation oracle.  It implements the
attached-flow subset of Riziotis & Voutsinas (2008), DOI 10.1002/fld.1525:

* body sources, one uniform bound-vorticity degree of freedom, material
  wake blobs, and one newborn trailing-edge vortex segment (their Eq. 3);
* the moving-wall condition ``(u - U_B) . n = w_T`` (their Eq. 4);
* an exactly closed, discrete Kelvin circulation ledger (the single-wake
  specialization of their Eq. 6);
* newborn length ``Delta S_W = mean(w_tau)_W Delta t`` (their Eq. 7); and
* emitted sheet strength from the two TE traces (their Eq. 8).

All integrated circulations in this module use the standard *counter-
clockwise-positive* point-vortex convention.  This is stated explicitly
because the cross-product convention in the source paper's printed Eq. 3 is
easy to read with the opposite sign.  With the clockwise body contour used
by :mod:`svi_dw_types`, positive angle of attack normally produces negative
CCW bound circulation and positive CCW starting-wake circulation.

S1 is not N2.6e1.  It has no integral-boundary-layer equations, no strong
viscous--inviscid coupling, no separation point, no second wake, no pressure
closure, and no force model.  A prescribed ``w_T`` is merely the typed
Neumann port that a later independently validated IBL stage may supply.
Passing the tests in this module therefore establishes only the attached
outer-flow gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .svi_dw_outer_2d import hess_smith_influence
from .svi_dw_types import ActualSurface2D, SVIDWValidationError


_TWO_PI = 2.0 * np.pi
_MAXIMUM_SYSTEM_CONDITION_NUMBER = 1.0e11
_GEOMETRY_RELATIVE_TOLERANCE = 2.0e-11
_MAXIMUM_GEOMETRY_ITERATIONS = 60
_NO_BIRTH_BRANCH_AGREEMENT_RELATIVE_TOLERANCE = 2.0e-10


class SVIUnsteadyOuterScopeError(NotImplementedError):
    """A caller requested physics beyond the attached S1 equation gate."""


class SVIUnsteadyOuterConvergenceError(RuntimeError):
    """The frozen-linear-system/newborn-geometry iteration did not converge."""


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


def _vector2(name: str, value: Any) -> np.ndarray:
    result = np.array(value, dtype=float, copy=True)
    if result.shape != (2,) or not np.all(np.isfinite(result)):
        raise SVIDWValidationError(f"{name} must be a finite vector with shape (2,)")
    result.setflags(write=False)
    return result


def _array(
    name: str,
    value: Any,
    shape: tuple[int, ...],
) -> np.ndarray:
    result = np.array(value, dtype=float, copy=True)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise SVIDWValidationError(
            f"{name} must be finite with shape {shape}, got {result.shape}"
        )
    result.setflags(write=False)
    return result


def _rotation(angle_rad: float) -> np.ndarray:
    cosine = np.cos(angle_rad)
    sine = np.sin(angle_rad)
    return np.array(((cosine, -sine), (sine, cosine)), dtype=float)


def _rotate_ccw_90(vector: np.ndarray) -> np.ndarray:
    return np.column_stack((-vector[..., 1], vector[..., 0]))


@dataclass(frozen=True)
class RigidKinematics2D:
    """Instantaneous rigid translation and pitch of a two-dimensional wall.

    ``pivot_body`` is fixed in body coordinates. ``pivot_inertial`` is its
    current inertial position.  The wall velocity is

    ``U_B = translation_velocity + omega k x (x - pivot_inertial)``.
    """

    pivot_body: Any = (0.0, 0.0)
    pivot_inertial: Any = (0.0, 0.0)
    angle_rad: float = 0.0
    translation_velocity_inertial: Any = (0.0, 0.0)
    angular_velocity_rad_s: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "pivot_body", _vector2("pivot_body", self.pivot_body))
        object.__setattr__(
            self,
            "pivot_inertial",
            _vector2("pivot_inertial", self.pivot_inertial),
        )
        object.__setattr__(
            self, "angle_rad", _finite_scalar("angle_rad", self.angle_rad)
        )
        object.__setattr__(
            self,
            "translation_velocity_inertial",
            _vector2(
                "translation_velocity_inertial",
                self.translation_velocity_inertial,
            ),
        )
        object.__setattr__(
            self,
            "angular_velocity_rad_s",
            _finite_scalar("angular_velocity_rad_s", self.angular_velocity_rad_s),
        )

    @property
    def rotation_body_to_inertial(self) -> np.ndarray:
        result = _rotation(self.angle_rad)
        result.setflags(write=False)
        return result

    def points_body_to_inertial(self, points: Any) -> np.ndarray:
        body = np.asarray(points, dtype=float)
        if body.ndim != 2 or body.shape[1] != 2 or not np.all(np.isfinite(body)):
            raise SVIDWValidationError("body points must be finite with shape (n,2)")
        result = (
            self.pivot_inertial
            + (body - self.pivot_body) @ self.rotation_body_to_inertial.T
        )
        result.setflags(write=False)
        return result

    def vectors_inertial_to_body(self, vectors: Any) -> np.ndarray:
        inertial = np.asarray(vectors, dtype=float)
        if inertial.shape[-1:] != (2,) or not np.all(np.isfinite(inertial)):
            raise SVIDWValidationError(
                "inertial vectors must be finite with final dimension 2"
            )
        result = inertial @ self.rotation_body_to_inertial
        result.setflags(write=False)
        return result

    def wall_velocity_body(self, body_points: Any) -> np.ndarray:
        points = np.asarray(body_points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
            raise SVIDWValidationError("body points must be finite with shape (n,2)")
        translation_body = self.vectors_inertial_to_body(
            self.translation_velocity_inertial
        )
        relative = points - self.pivot_body
        rotational = self.angular_velocity_rad_s * _rotate_ccw_90(relative)
        result = translation_body + rotational
        result.setflags(write=False)
        return result

    def wall_velocity_inertial(self, body_points: Any) -> np.ndarray:
        body_velocity = self.wall_velocity_body(body_points)
        result = body_velocity @ self.rotation_body_to_inertial.T
        result.setflags(write=False)
        return result


@dataclass(frozen=True)
class MaterialVortexBlob2D:
    """A material vortex blob with explicit CCW circulation and core.

    The regularized velocity is the Rosenhead form
    ``Gamma k x r / (2 pi (|r|^2 + core_radius^2))``.  The core is a
    declared numerical-resolution input; this module supplies no hidden core
    multiple or force-calibration rule.
    """

    position_inertial: Any
    circulation_ccw: float
    core_radius: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "position_inertial",
            _vector2("position_inertial", self.position_inertial),
        )
        object.__setattr__(
            self,
            "circulation_ccw",
            _finite_scalar("circulation_ccw", self.circulation_ccw),
        )
        object.__setattr__(
            self,
            "core_radius",
            _positive_scalar("core_radius", self.core_radius),
        )


@dataclass(frozen=True)
class TENearWakeSegment2D:
    """Frozen geometry of the current TE segment in body coordinates.

    The segment starts exactly at the trailing edge.  Its direction is one
    of the two outgoing wall tangents, as prescribed in the discussion after
    Riziotis--Voutsinas Eq. 8.  In the explicit CCW convention used here,
    negative predicted change of bound circulation selects the lower-side
    outgoing tangent; positive change selects the upper-side outgoing
    tangent.  The paper states the sign-dependent side rule but does not
    print this sign-to-side mapping, so this mapping is an explicit S1
    convention and must be tested under circulation-sign reversal.

    ``length = mean_emission_speed * time_step`` is Eq. 7 exactly.  Geometry
    is iterated against the solved mean TE trace by
    :func:`solve_attached_unsteady_outer_step`.
    """

    start_body: Any
    end_body: Any
    orientation_side: str
    mean_emission_speed: float
    time_step: float

    def __post_init__(self) -> None:
        start = _vector2("start_body", self.start_body)
        end = _vector2("end_body", self.end_body)
        if self.orientation_side not in {"upper", "lower"}:
            raise SVIDWValidationError("orientation_side must be 'upper' or 'lower'")
        speed = _positive_scalar("mean_emission_speed", self.mean_emission_speed)
        time_step = _positive_scalar("time_step", self.time_step)
        length = float(np.linalg.norm(end - start))
        coordinate_scale = max(
            float(np.max(np.abs(np.vstack((start, end))), initial=0.0)),
            1.0,
        )
        zero_tolerance = 256.0 * np.finfo(float).eps * coordinate_scale
        if length <= zero_tolerance:
            raise SVIDWValidationError("TE near-wake segment has zero numerical length")
        expected = speed * time_step
        equation_tolerance = (
            2048.0 * np.finfo(float).eps * max(length, expected, coordinate_scale)
        )
        if abs(length - expected) > equation_tolerance:
            raise SVIDWValidationError(
                "TE near-wake segment violates Eq. 7: "
                "length != mean_emission_speed * time_step"
            )
        object.__setattr__(self, "start_body", start)
        object.__setattr__(self, "end_body", end)
        object.__setattr__(self, "mean_emission_speed", speed)
        object.__setattr__(self, "time_step", time_step)

    @property
    def vector_body(self) -> np.ndarray:
        result = self.end_body - self.start_body
        result.setflags(write=False)
        return result

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.vector_body))

    @property
    def tangent_body(self) -> np.ndarray:
        result = self.vector_body / self.length
        result.setflags(write=False)
        return result


@dataclass(frozen=True)
class CommonTEDiagnostic2D:
    """The two adjacent-panel traces used by Riziotis Eq. 7--8.

    The source discretization evaluates the common trailing-edge velocity at
    the centres of the first lower and last upper panels.  No extrapolation
    to the cusp is hidden here.  Coordinates and centre-to-TE distances make
    that finite-resolution choice explicit and refinable.
    """

    trailing_edge_body: Any
    lower_panel_index: int
    lower_panel_center_body: Any
    lower_downstream_trace: float
    lower_te_distance: float
    lower_local_panel_length: float
    upper_panel_index: int
    upper_panel_center_body: Any
    upper_downstream_trace: float
    upper_te_distance: float
    upper_local_panel_length: float
    mean_downstream_trace: float
    jump_ccw: float

    def __post_init__(self) -> None:
        trailing_edge = _vector2("trailing_edge_body", self.trailing_edge_body)
        lower_center = _vector2("lower_panel_center_body", self.lower_panel_center_body)
        upper_center = _vector2("upper_panel_center_body", self.upper_panel_center_body)
        lower_index = int(self.lower_panel_index)
        upper_index = int(self.upper_panel_index)
        if lower_index != self.lower_panel_index or lower_index < 0:
            raise SVIDWValidationError(
                "lower_panel_index must be a non-negative integer"
            )
        if upper_index != self.upper_panel_index or upper_index < 0:
            raise SVIDWValidationError(
                "upper_panel_index must be a non-negative integer"
            )
        lower_trace = _positive_scalar(
            "lower_downstream_trace", self.lower_downstream_trace
        )
        upper_trace = _positive_scalar(
            "upper_downstream_trace", self.upper_downstream_trace
        )
        lower_distance = _positive_scalar("lower_te_distance", self.lower_te_distance)
        upper_distance = _positive_scalar("upper_te_distance", self.upper_te_distance)
        lower_length = _positive_scalar(
            "lower_local_panel_length", self.lower_local_panel_length
        )
        upper_length = _positive_scalar(
            "upper_local_panel_length", self.upper_local_panel_length
        )
        mean = _positive_scalar("mean_downstream_trace", self.mean_downstream_trace)
        jump = _finite_scalar("jump_ccw", self.jump_ccw)
        scale = max(lower_trace, upper_trace, mean, abs(jump), 1.0)
        tolerance = 4096.0 * np.finfo(float).eps * scale
        if abs(mean - 0.5 * (lower_trace + upper_trace)) > tolerance:
            raise SVIDWValidationError(
                "mean_downstream_trace is not the mean of the two TE traces"
            )
        if abs(jump - (lower_trace - upper_trace)) > tolerance:
            raise SVIDWValidationError(
                "jump_ccw is not lower_downstream_trace minus upper_downstream_trace"
            )
        object.__setattr__(self, "trailing_edge_body", trailing_edge)
        object.__setattr__(self, "lower_panel_index", lower_index)
        object.__setattr__(self, "lower_panel_center_body", lower_center)
        object.__setattr__(self, "lower_downstream_trace", lower_trace)
        object.__setattr__(self, "lower_te_distance", lower_distance)
        object.__setattr__(self, "lower_local_panel_length", lower_length)
        object.__setattr__(self, "upper_panel_index", upper_index)
        object.__setattr__(self, "upper_panel_center_body", upper_center)
        object.__setattr__(self, "upper_downstream_trace", upper_trace)
        object.__setattr__(self, "upper_te_distance", upper_distance)
        object.__setattr__(self, "upper_local_panel_length", upper_length)
        object.__setattr__(self, "mean_downstream_trace", mean)
        object.__setattr__(self, "jump_ccw", jump)

    @property
    def lower_te_distance_over_panel_length(self) -> float:
        return self.lower_te_distance / self.lower_local_panel_length

    @property
    def upper_te_distance_over_panel_length(self) -> float:
        return self.upper_te_distance / self.upper_local_panel_length


def te_orientation_side_from_ccw_bound_change(
    predicted_bound_circulation_change_ccw: float,
) -> str:
    """Apply S1's explicit sign-to-wall-side TE orientation convention."""
    change = _finite_scalar(
        "predicted_bound_circulation_change_ccw",
        predicted_bound_circulation_change_ccw,
    )
    return "lower" if change <= 0.0 else "upper"


def build_te_near_wake_segment(
    surface: ActualSurface2D,
    *,
    predicted_bound_circulation_change_ccw: float,
    mean_emission_speed: float,
    time_step: float,
) -> TENearWakeSegment2D:
    """Build a TE segment satisfying the Eq. 7 length identity."""
    if not isinstance(surface, ActualSurface2D):
        raise SVIDWValidationError("surface must be an ActualSurface2D")
    side = te_orientation_side_from_ccw_bound_change(
        predicted_bound_circulation_change_ccw
    )
    return _build_te_near_wake_segment_for_side(
        surface,
        orientation_side=side,
        mean_emission_speed=mean_emission_speed,
        time_step=time_step,
    )


def _build_te_near_wake_segment_for_side(
    surface: ActualSurface2D,
    *,
    orientation_side: str,
    mean_emission_speed: float,
    time_step: float,
) -> TENearWakeSegment2D:
    """Build one explicitly requested orientation for internal branch audit."""
    if not isinstance(surface, ActualSurface2D):
        raise SVIDWValidationError("surface must be an ActualSurface2D")
    if orientation_side not in {"upper", "lower"}:
        raise SVIDWValidationError("orientation_side must be 'upper' or 'lower'")
    side = orientation_side
    start = surface.upper_nodes[-1]
    if side == "upper":
        direction = surface.panel_tangents[-1]
    else:
        direction = -surface.panel_tangents[0]
    direction = direction / np.linalg.norm(direction)
    speed = _positive_scalar("mean_emission_speed", mean_emission_speed)
    delta_t = _positive_scalar("time_step", time_step)
    end = start + direction * speed * delta_t
    return TENearWakeSegment2D(
        start_body=start,
        end_body=end,
        orientation_side=side,
        mean_emission_speed=speed,
        time_step=delta_t,
    )


@dataclass(frozen=True)
class AttachedOuterStepInput2D:
    """Typed inputs for one attached S1 outer-flow time step."""

    surface: ActualSurface2D
    kinematics: RigidKinematics2D
    freestream_velocity_inertial: Any
    time_step: float
    previous_bound_circulation_ccw: float
    predicted_bound_circulation_change_ccw: float
    old_blobs: tuple[MaterialVortexBlob2D, ...] = ()
    wall_transpiration_velocity: Any | None = None
    initial_mean_emission_speed: float | None = None
    kelvin_reference_total_ccw: float | None = None
    stage_time: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.surface, ActualSurface2D):
            raise SVIDWValidationError("surface must be an ActualSurface2D")
        if not isinstance(self.kinematics, RigidKinematics2D):
            raise SVIDWValidationError("kinematics must be RigidKinematics2D")
        object.__setattr__(
            self,
            "freestream_velocity_inertial",
            _vector2(
                "freestream_velocity_inertial",
                self.freestream_velocity_inertial,
            ),
        )
        object.__setattr__(
            self, "time_step", _positive_scalar("time_step", self.time_step)
        )
        object.__setattr__(
            self, "stage_time", _finite_scalar("stage_time", self.stage_time)
        )
        object.__setattr__(
            self,
            "previous_bound_circulation_ccw",
            _finite_scalar(
                "previous_bound_circulation_ccw",
                self.previous_bound_circulation_ccw,
            ),
        )
        object.__setattr__(
            self,
            "predicted_bound_circulation_change_ccw",
            _finite_scalar(
                "predicted_bound_circulation_change_ccw",
                self.predicted_bound_circulation_change_ccw,
            ),
        )
        blobs = tuple(self.old_blobs)
        if not all(isinstance(item, MaterialVortexBlob2D) for item in blobs):
            raise SVIDWValidationError(
                "old_blobs must contain only MaterialVortexBlob2D"
            )
        object.__setattr__(self, "old_blobs", blobs)
        inventory_total = self.previous_bound_circulation_ccw + float(
            sum(item.circulation_ccw for item in blobs)
        )
        if self.kelvin_reference_total_ccw is not None:
            reference = _finite_scalar(
                "kelvin_reference_total_ccw",
                self.kelvin_reference_total_ccw,
            )
            scale = max(abs(reference), abs(inventory_total), 1.0)
            tolerance = 4096.0 * np.finfo(float).eps * scale
            if abs(inventory_total - reference) > tolerance:
                raise SVIDWValidationError(
                    "incoming bound-plus-material circulation does not "
                    "match kelvin_reference_total_ccw"
                )
            object.__setattr__(self, "kelvin_reference_total_ccw", reference)
        if self.wall_transpiration_velocity is None:
            transpiration = np.zeros(self.surface.panel_count, dtype=float)
            transpiration.setflags(write=False)
        else:
            transpiration = _array(
                "wall_transpiration_velocity",
                self.wall_transpiration_velocity,
                (self.surface.panel_count,),
            )
        object.__setattr__(self, "wall_transpiration_velocity", transpiration)
        if self.initial_mean_emission_speed is not None:
            object.__setattr__(
                self,
                "initial_mean_emission_speed",
                _positive_scalar(
                    "initial_mean_emission_speed",
                    self.initial_mean_emission_speed,
                ),
            )

    @classmethod
    def from_history(
        cls,
        *,
        surface: ActualSurface2D,
        kinematics: RigidKinematics2D,
        freestream_velocity_inertial: Any,
        time_step: float,
        history: "AttachedOuterHistory2D",
        predicted_bound_circulation_change_ccw: float,
        wall_transpiration_velocity: Any | None = None,
        initial_mean_emission_speed: float | None = None,
    ) -> "AttachedOuterStepInput2D":
        """Construct a step input without dropping the Kelvin reference."""
        if not isinstance(history, AttachedOuterHistory2D):
            raise SVIDWValidationError("history must be AttachedOuterHistory2D")
        return cls(
            surface=surface,
            kinematics=kinematics,
            freestream_velocity_inertial=freestream_velocity_inertial,
            time_step=time_step,
            previous_bound_circulation_ccw=history.bound_circulation_ccw,
            predicted_bound_circulation_change_ccw=(
                predicted_bound_circulation_change_ccw
            ),
            old_blobs=history.material_blobs,
            wall_transpiration_velocity=wall_transpiration_velocity,
            initial_mean_emission_speed=initial_mean_emission_speed,
            kelvin_reference_total_ccw=history.kelvin_reference_total_ccw,
            stage_time=history.stage_time,
        )


@dataclass(frozen=True)
class AttachedOuterStepSolution2D:
    """Auditable S1 solution; it intentionally contains no pressure or force."""

    inputs: AttachedOuterStepInput2D
    near_wake_segment: TENearWakeSegment2D
    source_strength: np.ndarray
    bound_sheet_strength_ccw: float
    newborn_sheet_strength_ccw: float
    bound_circulation_ccw: float
    newborn_circulation_ccw: float
    old_blob_circulation_ccw: float
    wall_velocity_body: np.ndarray
    relative_surface_velocity_body: np.ndarray
    relative_tangential_velocity: np.ndarray
    normal_boundary_residual: np.ndarray
    common_te_diagnostic: CommonTEDiagnostic2D
    emission_jump_ccw: float
    emission_residual: float
    mean_emission_speed: float
    eq7_length_residual: float
    kelvin_initial_total_ccw: float
    kelvin_final_total_ccw: float
    kelvin_residual: float
    linear_system_residual: np.ndarray
    linear_system_condition_number: float
    geometry_iterations: int

    @property
    def maximum_normal_boundary_residual(self) -> float:
        return float(np.max(np.abs(self.normal_boundary_residual), initial=0.0))


@dataclass(frozen=True)
class AttachedOuterHistory2D:
    """Cross-step bound/material inventory with a conserved Kelvin reference."""

    bound_circulation_ccw: float
    material_blobs: tuple[MaterialVortexBlob2D, ...]
    kelvin_reference_total_ccw: float
    stage_time: float

    def __post_init__(self) -> None:
        bound = _finite_scalar("bound_circulation_ccw", self.bound_circulation_ccw)
        blobs = tuple(self.material_blobs)
        if not all(isinstance(item, MaterialVortexBlob2D) for item in blobs):
            raise SVIDWValidationError(
                "material_blobs must contain only MaterialVortexBlob2D"
            )
        reference = _finite_scalar(
            "kelvin_reference_total_ccw",
            self.kelvin_reference_total_ccw,
        )
        stage_time = _finite_scalar("stage_time", self.stage_time)
        inventory_total = bound + float(sum(item.circulation_ccw for item in blobs))
        scale = max(abs(reference), abs(inventory_total), 1.0)
        tolerance = 4096.0 * np.finfo(float).eps * scale
        if abs(inventory_total - reference) > tolerance:
            raise SVIDWValidationError(
                "history violates its Kelvin reference: "
                "bound + material blobs != reference total"
            )
        object.__setattr__(self, "bound_circulation_ccw", bound)
        object.__setattr__(self, "material_blobs", blobs)
        object.__setattr__(self, "kelvin_reference_total_ccw", reference)
        object.__setattr__(self, "stage_time", stage_time)


@dataclass(frozen=True)
class MaterialAdvectionDiagnostic2D:
    """Internally assembled material-blob velocities for one Euler march."""

    position_inertial: Any
    freestream_velocity_inertial: Any
    body_panel_velocity_inertial: Any
    other_material_blob_velocity_inertial: Any
    current_near_wake_velocity_inertial: Any
    total_velocity_inertial: Any
    newborn_index: int

    def __post_init__(self) -> None:
        positions = np.asarray(self.position_inertial, dtype=float)
        if positions.ndim != 2 or positions.shape[1:] != (2,):
            raise SVIDWValidationError("position_inertial must have shape (n,2)")
        count = positions.shape[0]
        arrays = {}
        for name in (
            "position_inertial",
            "freestream_velocity_inertial",
            "body_panel_velocity_inertial",
            "other_material_blob_velocity_inertial",
            "current_near_wake_velocity_inertial",
            "total_velocity_inertial",
        ):
            arrays[name] = _array(name, getattr(self, name), (count, 2))
        newborn_index = int(self.newborn_index)
        if newborn_index != self.newborn_index or newborn_index != count - 1:
            raise SVIDWValidationError(
                "newborn_index must identify the final material entry"
            )
        reconstructed = (
            arrays["freestream_velocity_inertial"]
            + arrays["body_panel_velocity_inertial"]
            + arrays["other_material_blob_velocity_inertial"]
            + arrays["current_near_wake_velocity_inertial"]
        )
        scale = max(
            float(
                np.max(
                    np.abs(arrays["total_velocity_inertial"]),
                    initial=0.0,
                )
            ),
            1.0,
        )
        tolerance = 4096.0 * np.finfo(float).eps * scale
        if (
            np.max(
                np.abs(arrays["total_velocity_inertial"] - reconstructed),
                initial=0.0,
            )
            > tolerance
        ):
            raise SVIDWValidationError(
                "material-advection velocity components do not sum to total"
            )
        if np.any(arrays["current_near_wake_velocity_inertial"][newborn_index]):
            raise SVIDWValidationError(
                "the newborn segment must not self-induce its collapsed blob"
            )
        for name, value in arrays.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "newborn_index", newborn_index)


@dataclass(frozen=True)
class AttachedOuterMarchResult2D:
    """One owned order: solve, birth, internally advect, advance history."""

    solution: AttachedOuterStepSolution2D
    history_at_stage_after_birth: AttachedOuterHistory2D
    advection: MaterialAdvectionDiagnostic2D
    history_next: AttachedOuterHistory2D

    def __post_init__(self) -> None:
        if not isinstance(self.solution, AttachedOuterStepSolution2D):
            raise SVIDWValidationError("solution must be AttachedOuterStepSolution2D")
        if not isinstance(self.history_at_stage_after_birth, AttachedOuterHistory2D):
            raise SVIDWValidationError(
                "history_at_stage_after_birth must be AttachedOuterHistory2D"
            )
        if not isinstance(self.advection, MaterialAdvectionDiagnostic2D):
            raise SVIDWValidationError(
                "advection must be MaterialAdvectionDiagnostic2D"
            )
        if not isinstance(self.history_next, AttachedOuterHistory2D):
            raise SVIDWValidationError("history_next must be AttachedOuterHistory2D")
        born = self.history_at_stage_after_birth
        advanced = self.history_next
        delta_t = self.solution.inputs.time_step
        time_scale = max(abs(born.stage_time), abs(advanced.stage_time), delta_t, 1.0)
        tolerance = 4096.0 * np.finfo(float).eps * time_scale
        if abs(born.stage_time - self.solution.inputs.stage_time) > tolerance:
            raise SVIDWValidationError("birth history is not at the solved stage")
        if abs(advanced.stage_time - (born.stage_time + delta_t)) > tolerance:
            raise SVIDWValidationError(
                "advanced history is not at stage_time + time_step"
            )
        if len(born.material_blobs) != len(advanced.material_blobs):
            raise SVIDWValidationError("material inventory changed during convection")
        if len(born.material_blobs) != self.advection.position_inertial.shape[0]:
            raise SVIDWValidationError(
                "advection diagnostic does not cover every material blob"
            )
        for index, (before, after) in enumerate(
            zip(born.material_blobs, advanced.material_blobs)
        ):
            if (
                before.circulation_ccw != after.circulation_ccw
                or before.core_radius != after.core_radius
            ):
                raise SVIDWValidationError(
                    "circulation or core changed during material convection"
                )
            expected = (
                before.position_inertial
                + delta_t * self.advection.total_velocity_inertial[index]
            )
            position_scale = max(float(np.max(np.abs(expected), initial=0.0)), 1.0)
            if (
                np.max(np.abs(after.position_inertial - expected), initial=0.0)
                > 4096.0 * np.finfo(float).eps * position_scale
            ):
                raise SVIDWValidationError(
                    "history_next violates explicit-Euler convection"
                )


def _vortex_blob_velocity_inertial(
    points_inertial: np.ndarray,
    blob: MaterialVortexBlob2D,
) -> np.ndarray:
    relative = points_inertial - blob.position_inertial
    radius_squared = np.einsum("ij,ij->i", relative, relative)
    denominator = _TWO_PI * (radius_squared + blob.core_radius**2)
    return blob.circulation_ccw * _rotate_ccw_90(relative) / denominator[:, None]


def material_blob_velocity_inertial(
    points_inertial: Any,
    blobs: Iterable[MaterialVortexBlob2D],
) -> np.ndarray:
    """Velocity induced by explicit-core material blobs."""
    points = np.asarray(points_inertial, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
        raise SVIDWValidationError("points_inertial must be finite with shape (n,2)")
    result = np.zeros_like(points)
    for blob in tuple(blobs):
        if not isinstance(blob, MaterialVortexBlob2D):
            raise SVIDWValidationError("blobs must contain only MaterialVortexBlob2D")
        result += _vortex_blob_velocity_inertial(points, blob)
    result.setflags(write=False)
    return result


def constant_vortex_segment_velocity_body(
    points_body: Any,
    segment: TENearWakeSegment2D,
) -> np.ndarray:
    """Velocity per unit CCW sheet strength of one straight segment."""
    points = np.asarray(points_body, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
        raise SVIDWValidationError("points_body must be finite with shape (n,2)")
    if not isinstance(segment, TENearWakeSegment2D):
        raise SVIDWValidationError("segment must be TENearWakeSegment2D")
    tangent = segment.tangent_body
    normal = np.array((-tangent[1], tangent[0]), dtype=float)
    relative = points - segment.start_body
    x_local = relative @ tangent
    y_local = relative @ normal
    x_after = x_local - segment.length
    radius_start_squared = x_local**2 + y_local**2
    radius_end_squared = x_after**2 + y_local**2
    coordinate_scale = max(
        segment.length,
        float(np.max(np.abs(points), initial=0.0)),
        1.0,
    )
    squared_tolerance = (256.0 * np.finfo(float).eps * coordinate_scale) ** 2
    if np.any(radius_start_squared <= squared_tolerance) or np.any(
        radius_end_squared <= squared_tolerance
    ):
        raise SVIDWValidationError(
            "a velocity target coincides with a near-wake endpoint"
        )
    source_tangent = 0.25 / np.pi * np.log(radius_start_squared / radius_end_squared)
    endpoint_angle = np.arctan2(
        y_local * segment.length,
        x_local * x_after + y_local**2,
    )
    source_normal = endpoint_angle / _TWO_PI
    # A positive CCW vortex distribution is the source velocity rotated
    # +90 degrees: (u_t,u_n)_vortex = (-u_n,u_t)_source.
    result = -source_normal[:, None] * tangent + source_tangent[:, None] * normal
    result.setflags(write=False)
    return result


def _body_panel_velocity_body(
    points_body: np.ndarray,
    *,
    surface: ActualSurface2D,
    source_strength: np.ndarray,
    bound_sheet_strength_ccw: float,
) -> np.ndarray:
    """Evaluate the just-solved source/bound field away from the wall."""
    points = np.asarray(points_body, dtype=float)
    if points.ndim != 2 or points.shape[1:] != (2,) or not np.all(np.isfinite(points)):
        raise SVIDWValidationError("points_body must be finite with shape (n,2)")
    source = np.asarray(source_strength, dtype=float)
    if source.shape != (surface.panel_count,) or not np.all(np.isfinite(source)):
        raise SVIDWValidationError(
            "source_strength must be finite with one value per panel"
        )
    bound = _finite_scalar("bound_sheet_strength_ccw", bound_sheet_strength_ccw)
    result = np.zeros_like(points)
    starts = surface.contour_nodes[:-1]
    for panel_index in range(surface.panel_count):
        tangent = surface.panel_tangents[panel_index]
        normal = surface.panel_outward_normals[panel_index]
        length = surface.panel_lengths[panel_index]
        relative = points - starts[panel_index]
        x_local = relative @ tangent
        y_local = relative @ normal
        x_after = x_local - length
        radius_start_squared = x_local**2 + y_local**2
        radius_end_squared = x_after**2 + y_local**2
        coordinate_scale = max(
            float(length),
            float(np.max(np.abs(points), initial=0.0)),
            1.0,
        )
        squared_tolerance = (256.0 * np.finfo(float).eps * coordinate_scale) ** 2
        if np.any(radius_start_squared <= squared_tolerance) or np.any(
            radius_end_squared <= squared_tolerance
        ):
            raise SVIDWValidationError(
                "a material-blob target coincides with a body-panel endpoint"
            )
        source_tangent = (
            0.25 / np.pi * np.log(radius_start_squared / radius_end_squared)
        )
        endpoint_angle = np.arctan2(
            y_local * length,
            x_local * x_after + y_local**2,
        )
        source_normal = endpoint_angle / _TWO_PI
        source_velocity = (
            source_tangent[:, None] * tangent + source_normal[:, None] * normal
        )
        vortex_velocity = (
            -source_normal[:, None] * tangent + source_tangent[:, None] * normal
        )
        result += source[panel_index] * source_velocity
        result += bound * vortex_velocity
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class _FrozenGeometrySolve:
    source_strength: np.ndarray
    bound_sheet_strength_ccw: float
    newborn_sheet_strength_ccw: float
    wall_velocity_body: np.ndarray
    relative_surface_velocity_body: np.ndarray
    relative_tangential_velocity: np.ndarray
    normal_residual: np.ndarray
    common_te_diagnostic: CommonTEDiagnostic2D
    emission_jump_ccw: float
    emission_residual: float
    mean_emission_speed: float
    linear_residual: np.ndarray
    condition_number: float


def _solve_frozen_near_wake(
    inputs: AttachedOuterStepInput2D,
    segment: TENearWakeSegment2D,
) -> _FrozenGeometrySolve:
    surface = inputs.surface
    n_panel = surface.panel_count
    influence = hess_smith_influence(surface)
    motion = inputs.kinematics
    panel_points_inertial = motion.points_body_to_inertial(surface.panel_midpoints)
    old_velocity_inertial = material_blob_velocity_inertial(
        panel_points_inertial, inputs.old_blobs
    )
    old_velocity_body = motion.vectors_inertial_to_body(old_velocity_inertial)
    freestream_body = motion.vectors_inertial_to_body(
        inputs.freestream_velocity_inertial
    )
    wall_velocity_body = motion.wall_velocity_body(surface.panel_midpoints)
    external_relative_velocity = (
        freestream_body + old_velocity_body - wall_velocity_body
    )
    external_normal = np.einsum(
        "ij,ij->i",
        external_relative_velocity,
        surface.panel_outward_normals,
    )
    external_tangential = np.einsum(
        "ij,ij->i",
        external_relative_velocity,
        surface.panel_tangents,
    )
    near_wake_velocity = constant_vortex_segment_velocity_body(
        surface.panel_midpoints, segment
    )
    near_wake_normal = np.einsum(
        "ij,ij->i",
        near_wake_velocity,
        surface.panel_outward_normals,
    )
    near_wake_tangential = np.einsum(
        "ij,ij->i",
        near_wake_velocity,
        surface.panel_tangents,
    )

    # Unknown order is exactly {sigma_1 ... sigma_N, gamma_B, gamma_W}.
    system = np.zeros((n_panel + 2, n_panel + 2), dtype=float)
    right_hand_side = np.zeros(n_panel + 2, dtype=float)
    system[:n_panel, :n_panel] = influence.source_normal
    system[:n_panel, n_panel] = influence.circulation_normal
    system[:n_panel, n_panel + 1] = near_wake_normal
    right_hand_side[:n_panel] = inputs.wall_transpiration_velocity - external_normal

    lower_te = 0
    upper_te = n_panel - 1
    te_indices = np.array((lower_te, upper_te), dtype=int)
    # Common downstream traces are
    #   w_lower = -w_t(clockwise lower panel),
    #   w_upper = +w_t(clockwise upper panel).
    # Riziotis Eq. 8 uses the paper's clockwise-positive sheet convention.
    # Our kernels use physical counter-clockwise-positive circulation, so
    # the converted emitted sheet strength is
    #   gamma_W,ccw = w_lower - w_upper
    #               = -(w_t,upper + w_t,lower).
    system[n_panel, :n_panel] = np.sum(influence.source_tangential[te_indices], axis=0)
    system[n_panel, n_panel] = float(
        np.sum(influence.circulation_tangential[te_indices])
    )
    system[n_panel, n_panel + 1] = float(np.sum(near_wake_tangential[te_indices])) + 1.0
    right_hand_side[n_panel] = -float(np.sum(external_tangential[te_indices]))

    # Old material blobs occur unchanged on both sides of the step ledger,
    # so exact Kelvin conservation reduces to
    #   Gamma_B,new + Gamma_W,new = Gamma_B,previous.
    # gamma_B and gamma_W are sheet strengths; perimeter and segment length
    # convert them to integrated CCW circulation.
    system[n_panel + 1, n_panel] = surface.perimeter
    system[n_panel + 1, n_panel + 1] = segment.length
    right_hand_side[n_panel + 1] = inputs.previous_bound_circulation_ccw

    condition_number = float(np.linalg.cond(system))
    if (
        not np.isfinite(condition_number)
        or condition_number > _MAXIMUM_SYSTEM_CONDITION_NUMBER
    ):
        raise SVIUnsteadyOuterConvergenceError(
            "S1 source/bound/newborn system is singular or too "
            f"ill-conditioned ({condition_number:.6e})"
        )
    try:
        unknown = np.linalg.solve(system, right_hand_side)
    except np.linalg.LinAlgError as error:
        raise SVIUnsteadyOuterConvergenceError(
            "S1 source/bound/newborn linear solve failed"
        ) from error

    source_strength = unknown[:n_panel]
    bound_strength = float(unknown[n_panel])
    newborn_strength = float(unknown[n_panel + 1])
    source_normal = influence.source_normal @ source_strength
    source_tangential = influence.source_tangential @ source_strength
    total_normal = (
        external_normal
        + source_normal
        + influence.circulation_normal * bound_strength
        + near_wake_normal * newborn_strength
    )
    total_tangential = (
        external_tangential
        + source_tangential
        + influence.circulation_tangential * bound_strength
        + near_wake_tangential * newborn_strength
    )
    relative_velocity = (
        total_tangential[:, None] * surface.panel_tangents
        + total_normal[:, None] * surface.panel_outward_normals
    )
    normal_residual = total_normal - inputs.wall_transpiration_velocity
    lower_common_speed = -float(total_tangential[lower_te])
    upper_common_speed = float(total_tangential[upper_te])
    if lower_common_speed <= 0.0 or upper_common_speed <= 0.0:
        raise SVIUnsteadyOuterScopeError(
            "attached S1 requires both adjacent-panel TE traces to be "
            "strictly downstream-positive; got "
            f"lower={lower_common_speed:.16e}, "
            f"upper={upper_common_speed:.16e}"
        )
    mean_speed = 0.5 * (lower_common_speed + upper_common_speed)
    emission_jump = lower_common_speed - upper_common_speed
    emission_residual = emission_jump - newborn_strength
    linear_residual = system @ unknown - right_hand_side
    trailing_edge = surface.upper_nodes[-1]
    common_te_diagnostic = CommonTEDiagnostic2D(
        trailing_edge_body=trailing_edge,
        lower_panel_index=lower_te,
        lower_panel_center_body=surface.panel_midpoints[lower_te],
        lower_downstream_trace=lower_common_speed,
        lower_te_distance=float(
            np.linalg.norm(surface.panel_midpoints[lower_te] - trailing_edge)
        ),
        lower_local_panel_length=float(surface.panel_lengths[lower_te]),
        upper_panel_index=upper_te,
        upper_panel_center_body=surface.panel_midpoints[upper_te],
        upper_downstream_trace=upper_common_speed,
        upper_te_distance=float(
            np.linalg.norm(surface.panel_midpoints[upper_te] - trailing_edge)
        ),
        upper_local_panel_length=float(surface.panel_lengths[upper_te]),
        mean_downstream_trace=mean_speed,
        jump_ccw=emission_jump,
    )

    return _FrozenGeometrySolve(
        source_strength=_array("source_strength", source_strength, (n_panel,)),
        bound_sheet_strength_ccw=bound_strength,
        newborn_sheet_strength_ccw=newborn_strength,
        wall_velocity_body=_array(
            "wall_velocity_body", wall_velocity_body, (n_panel, 2)
        ),
        relative_surface_velocity_body=_array(
            "relative_surface_velocity_body",
            relative_velocity,
            (n_panel, 2),
        ),
        relative_tangential_velocity=_array(
            "relative_tangential_velocity",
            total_tangential,
            (n_panel,),
        ),
        normal_residual=_array("normal_residual", normal_residual, (n_panel,)),
        common_te_diagnostic=common_te_diagnostic,
        emission_jump_ccw=float(emission_jump),
        emission_residual=float(emission_residual),
        mean_emission_speed=float(mean_speed),
        linear_residual=_array("linear_residual", linear_residual, (n_panel + 2,)),
        condition_number=condition_number,
    )


def _initial_mean_emission_speed(
    inputs: AttachedOuterStepInput2D,
) -> float:
    if inputs.initial_mean_emission_speed is not None:
        return inputs.initial_mean_emission_speed
    surface = inputs.surface
    motion = inputs.kinematics
    te_indices = np.array((0, surface.panel_count - 1), dtype=int)
    body_points = surface.panel_midpoints[te_indices]
    inertial_points = motion.points_body_to_inertial(body_points)
    old_inertial = material_blob_velocity_inertial(inertial_points, inputs.old_blobs)
    relative_body = motion.vectors_inertial_to_body(
        inputs.freestream_velocity_inertial + old_inertial
    ) - motion.wall_velocity_body(body_points)
    contour_trace = np.einsum(
        "ij,ij->i",
        relative_body,
        surface.panel_tangents[te_indices],
    )
    mean_speed = 0.5 * (-contour_trace[0] + contour_trace[1])
    return _positive_scalar("initial mean TE emission speed", mean_speed)


@dataclass(frozen=True)
class _OrientationBranchSolve:
    orientation_side: str
    frozen: _FrozenGeometrySolve
    segment: TENearWakeSegment2D
    geometry_iterations: int
    bound_circulation_ccw: float
    newborn_circulation_ccw: float
    bound_circulation_change_ccw: float
    sign_tolerance: float

    @property
    def is_roundoff_no_birth(self) -> bool:
        return abs(self.bound_circulation_change_ccw) <= self.sign_tolerance

    @property
    def is_nonzero_sign_consistent(self) -> bool:
        change = self.bound_circulation_change_ccw
        if self.orientation_side == "lower":
            return change < -self.sign_tolerance
        return change > self.sign_tolerance


def _solve_orientation_branch(
    inputs: AttachedOuterStepInput2D,
    orientation_side: str,
) -> _OrientationBranchSolve:
    """Converge Eq. 7 for one branch without treating it as physical yet."""
    mean_speed = _initial_mean_emission_speed(inputs)
    frozen: _FrozenGeometrySolve | None = None
    segment: TENearWakeSegment2D | None = None
    geometry_iterations = 0
    for geometry_iterations in range(1, _MAXIMUM_GEOMETRY_ITERATIONS + 1):
        segment = _build_te_near_wake_segment_for_side(
            inputs.surface,
            orientation_side=orientation_side,
            mean_emission_speed=mean_speed,
            time_step=inputs.time_step,
        )
        frozen = _solve_frozen_near_wake(inputs, segment)
        solved_speed = frozen.mean_emission_speed
        relative_change = abs(solved_speed - mean_speed) / max(
            solved_speed,
            mean_speed,
            np.finfo(float).tiny,
        )
        if relative_change <= _GEOMETRY_RELATIVE_TOLERANCE:
            break
        mean_speed = solved_speed
    else:
        raise SVIUnsteadyOuterConvergenceError(
            f"{orientation_side} Eq. 7 near-wake geometry did not "
            "converge in "
            f"{_MAXIMUM_GEOMETRY_ITERATIONS} fixed-point iterations"
        )
    assert frozen is not None
    assert segment is not None
    bound_circulation = frozen.bound_sheet_strength_ccw * inputs.surface.perimeter
    newborn_circulation = frozen.newborn_sheet_strength_ccw * segment.length
    actual_change = bound_circulation - inputs.previous_bound_circulation_ccw
    circulation_scale = max(
        abs(bound_circulation),
        abs(inputs.previous_bound_circulation_ccw),
        abs(newborn_circulation),
        1.0,
    )
    sign_tolerance = 4096.0 * np.finfo(float).eps * circulation_scale
    return _OrientationBranchSolve(
        orientation_side=orientation_side,
        frozen=frozen,
        segment=segment,
        geometry_iterations=geometry_iterations,
        bound_circulation_ccw=float(bound_circulation),
        newborn_circulation_ccw=float(newborn_circulation),
        bound_circulation_change_ccw=float(actual_change),
        sign_tolerance=float(sign_tolerance),
    )


def _no_birth_branch_solutions_agree(
    lower: _OrientationBranchSolve,
    upper: _OrientationBranchSolve,
) -> bool:
    """Require physical fields, not merely signs, to share the zero-birth limit."""

    def arrays_agree(left: Any, right: Any) -> bool:
        left_array = np.asarray(left, dtype=float)
        right_array = np.asarray(right, dtype=float)
        if left_array.shape != right_array.shape:
            return False
        scale = max(
            float(np.max(np.abs(left_array), initial=0.0)),
            float(np.max(np.abs(right_array), initial=0.0)),
            1.0,
        )
        tolerance = (
            _NO_BIRTH_BRANCH_AGREEMENT_RELATIVE_TOLERANCE * scale
            + 4096.0 * np.finfo(float).eps * scale
        )
        return bool(np.max(np.abs(left_array - right_array), initial=0.0) <= tolerance)

    lower_frozen = lower.frozen
    upper_frozen = upper.frozen
    comparisons = (
        (lower_frozen.source_strength, upper_frozen.source_strength),
        (
            lower_frozen.relative_surface_velocity_body,
            upper_frozen.relative_surface_velocity_body,
        ),
        (
            lower_frozen.relative_tangential_velocity,
            upper_frozen.relative_tangential_velocity,
        ),
        (
            lower_frozen.bound_sheet_strength_ccw,
            upper_frozen.bound_sheet_strength_ccw,
        ),
        (
            lower_frozen.newborn_sheet_strength_ccw,
            upper_frozen.newborn_sheet_strength_ccw,
        ),
        (
            lower_frozen.mean_emission_speed,
            upper_frozen.mean_emission_speed,
        ),
        (
            lower_frozen.common_te_diagnostic.lower_downstream_trace,
            upper_frozen.common_te_diagnostic.lower_downstream_trace,
        ),
        (
            lower_frozen.common_te_diagnostic.upper_downstream_trace,
            upper_frozen.common_te_diagnostic.upper_downstream_trace,
        ),
        (lower.segment.length, upper.segment.length),
    )
    return all(arrays_agree(left, right) for left, right in comparisons)


def _select_physical_orientation_branch(
    branches: dict[str, _OrientationBranchSolve],
    branch_errors: dict[str, BaseException],
) -> _OrientationBranchSolve:
    """Apply the fail-closed two-branch orientation rule."""
    consistent = [
        branch for branch in branches.values() if branch.is_nonzero_sign_consistent
    ]
    if len(consistent) == 1:
        return consistent[0]
    if len(consistent) == 2:
        raise SVIUnsteadyOuterConvergenceError(
            "both nonzero TE orientation branches are sign-consistent; "
            "the attached S1 branch is ambiguous and was rejected"
        )
    lower = branches.get("lower")
    upper = branches.get("upper")
    if (
        lower is not None
        and upper is not None
        and lower.is_roundoff_no_birth
        and upper.is_roundoff_no_birth
    ):
        if not _no_birth_branch_solutions_agree(lower, upper):
            raise SVIUnsteadyOuterConvergenceError(
                "both TE branches have roundoff-level birth, but their "
                "solved fields do not share a common no-birth limit"
            )
        return lower
    details = ", ".join(
        f"{side}={type(error).__name__}: {error}"
        for side, error in sorted(branch_errors.items())
    )
    changes = ", ".join(
        f"{side} DeltaGamma_B={branch.bound_circulation_change_ccw:.16e}"
        for side, branch in sorted(branches.items())
    )
    suffix = "; ".join(item for item in (changes, details) if item)
    raise SVIUnsteadyOuterConvergenceError(
        "zero nonzero TE orientation branches are sign-consistent; "
        "the attached S1 branch was rejected" + (f" ({suffix})" if suffix else "")
    )


def solve_attached_unsteady_outer_step(
    inputs: AttachedOuterStepInput2D,
) -> AttachedOuterStepSolution2D:
    """Solve one S1 attached outer-flow step.

    Both source-method TE orientations are solved internally.  The caller's
    circulation-change predictor only controls evaluation order; it cannot
    select the accepted branch.  Exactly one nonzero branch must agree with
    its solved bound-circulation-change sign.  The exact/roundoff no-birth
    limit is deterministically lower only after both solved fields agree.

    The small outer iteration updates only the Eq. 7 newborn length.  It is
    not a viscous--inviscid iteration.
    """
    if not isinstance(inputs, AttachedOuterStepInput2D):
        raise SVIDWValidationError("inputs must be AttachedOuterStepInput2D")
    preferred = te_orientation_side_from_ccw_bound_change(
        inputs.predicted_bound_circulation_change_ccw
    )
    order = (preferred, "upper" if preferred == "lower" else "lower")
    branches: dict[str, _OrientationBranchSolve] = {}
    branch_errors: dict[str, BaseException] = {}
    for side in order:
        try:
            branches[side] = _solve_orientation_branch(inputs, side)
        except (
            SVIUnsteadyOuterConvergenceError,
            SVIUnsteadyOuterScopeError,
            SVIDWValidationError,
        ) as error:
            branch_errors[side] = error
    selected = _select_physical_orientation_branch(branches, branch_errors)
    frozen = selected.frozen
    segment = selected.segment
    bound_circulation = selected.bound_circulation_ccw
    newborn_circulation = selected.newborn_circulation_ccw
    old_circulation = float(sum(blob.circulation_ccw for blob in inputs.old_blobs))
    kelvin_initial = inputs.previous_bound_circulation_ccw + old_circulation
    kelvin_final = bound_circulation + newborn_circulation + old_circulation
    kelvin_residual = kelvin_final - kelvin_initial
    eq7_residual = segment.length - frozen.mean_emission_speed * inputs.time_step

    return AttachedOuterStepSolution2D(
        inputs=inputs,
        near_wake_segment=segment,
        source_strength=frozen.source_strength,
        bound_sheet_strength_ccw=frozen.bound_sheet_strength_ccw,
        newborn_sheet_strength_ccw=frozen.newborn_sheet_strength_ccw,
        bound_circulation_ccw=float(bound_circulation),
        newborn_circulation_ccw=float(newborn_circulation),
        old_blob_circulation_ccw=old_circulation,
        wall_velocity_body=frozen.wall_velocity_body,
        relative_surface_velocity_body=(frozen.relative_surface_velocity_body),
        relative_tangential_velocity=(frozen.relative_tangential_velocity),
        normal_boundary_residual=frozen.normal_residual,
        common_te_diagnostic=frozen.common_te_diagnostic,
        emission_jump_ccw=frozen.emission_jump_ccw,
        emission_residual=frozen.emission_residual,
        mean_emission_speed=frozen.mean_emission_speed,
        eq7_length_residual=float(eq7_residual),
        kelvin_initial_total_ccw=kelvin_initial,
        kelvin_final_total_ccw=kelvin_final,
        kelvin_residual=float(kelvin_residual),
        linear_system_residual=frozen.linear_residual,
        linear_system_condition_number=frozen.condition_number,
        geometry_iterations=selected.geometry_iterations,
    )


def completed_step_to_material_blob(
    solution: AttachedOuterStepSolution2D,
    *,
    core_radius: float,
) -> MaterialVortexBlob2D:
    """Collapse the solved newborn segment without accepting a free Gamma.

    The birth position uses the same kinematic stage that defined and solved
    the near-wake segment.  Integrated circulation is uniquely inherited from
    ``newborn_sheet_strength_ccw * segment.length`` as stored by the solution;
    a caller cannot inject an independently signed or scaled value.
    """
    if not isinstance(solution, AttachedOuterStepSolution2D):
        raise SVIDWValidationError("solution must be AttachedOuterStepSolution2D")
    segment = solution.near_wake_segment
    kinematics = solution.inputs.kinematics
    midpoint_body = 0.5 * (segment.start_body + segment.end_body)
    midpoint_inertial = kinematics.points_body_to_inertial(midpoint_body[None, :])[0]
    return MaterialVortexBlob2D(
        position_inertial=midpoint_inertial,
        circulation_ccw=solution.newborn_circulation_ccw,
        core_radius=core_radius,
    )


def complete_attached_outer_history(
    solution: AttachedOuterStepSolution2D,
    *,
    newborn_core_radius: float,
) -> AttachedOuterHistory2D:
    """Append one solved birth while preserving the global Kelvin invariant.

    Old blobs are already located at ``solution.inputs.stage_time`` because
    they induced the just-solved flow.  They are copied without accepting
    caller-supplied replacement positions.  Material convection to the next
    stage is a separate, explicitly named operation below.
    """
    if not isinstance(solution, AttachedOuterStepSolution2D):
        raise SVIDWValidationError("solution must be AttachedOuterStepSolution2D")
    if solution.inputs.kelvin_reference_total_ccw is None:
        reference = solution.kelvin_initial_total_ccw
    else:
        reference = solution.inputs.kelvin_reference_total_ccw
    scale = max(
        abs(reference),
        abs(solution.kelvin_final_total_ccw),
        1.0,
    )
    tolerance = 8192.0 * np.finfo(float).eps * scale
    if abs(solution.kelvin_final_total_ccw - reference) > tolerance:
        raise SVIDWValidationError(
            "solved final circulation drifted from the incoming Kelvin "
            "reference; the residual cannot be absorbed into new history"
        )
    newborn = completed_step_to_material_blob(solution, core_radius=newborn_core_radius)
    return AttachedOuterHistory2D(
        bound_circulation_ccw=solution.bound_circulation_ccw,
        material_blobs=solution.inputs.old_blobs + (newborn,),
        kelvin_reference_total_ccw=reference,
        stage_time=solution.inputs.stage_time,
    )


def _material_advection_from_completed_step(
    solution: AttachedOuterStepSolution2D,
    history_after_birth: AttachedOuterHistory2D,
) -> MaterialAdvectionDiagnostic2D:
    """Assemble stage-consistent velocities with every self term excluded."""
    if not isinstance(solution, AttachedOuterStepSolution2D):
        raise SVIDWValidationError("solution must be AttachedOuterStepSolution2D")
    if not isinstance(history_after_birth, AttachedOuterHistory2D):
        raise SVIDWValidationError("history_after_birth must be AttachedOuterHistory2D")
    old_blobs = solution.inputs.old_blobs
    blobs = history_after_birth.material_blobs
    if len(blobs) != len(old_blobs) + 1:
        raise SVIDWValidationError(
            "history_after_birth must append exactly the solved newborn"
        )
    count = len(blobs)
    newborn_index = count - 1
    positions = np.vstack([blob.position_inertial for blob in blobs])
    motion = solution.inputs.kinematics
    rotation = motion.rotation_body_to_inertial
    points_body = motion.pivot_body + (positions - motion.pivot_inertial) @ rotation
    body_velocity_body = _body_panel_velocity_body(
        points_body,
        surface=solution.inputs.surface,
        source_strength=solution.source_strength,
        bound_sheet_strength_ccw=solution.bound_sheet_strength_ccw,
    )
    body_velocity_inertial = body_velocity_body @ rotation.T
    freestream = np.broadcast_to(
        solution.inputs.freestream_velocity_inertial, (count, 2)
    ).copy()

    # Existing material blobs interact pairwise, excluding their own
    # Rosenhead kernel.  The current birth remains its solved finite segment
    # during this stage: old blobs see that segment, while the collapsed
    # newborn excludes it as its own field.  Thus it is never counted both as
    # a segment and as a point blob.
    other_blob_velocity = np.zeros((count, 2), dtype=float)
    for target_index in range(count):
        for source_index, source_blob in enumerate(old_blobs):
            if target_index == source_index:
                continue
            other_blob_velocity[target_index] += _vortex_blob_velocity_inertial(
                positions[target_index : target_index + 1],
                source_blob,
            )[0]
    near_wake_velocity = np.zeros((count, 2), dtype=float)
    if old_blobs:
        near_wake_body = (
            constant_vortex_segment_velocity_body(
                points_body[: len(old_blobs)],
                solution.near_wake_segment,
            )
            * solution.newborn_sheet_strength_ccw
        )
        near_wake_velocity[: len(old_blobs)] = near_wake_body @ rotation.T
    total = (
        freestream + body_velocity_inertial + other_blob_velocity + near_wake_velocity
    )
    return MaterialAdvectionDiagnostic2D(
        position_inertial=positions,
        freestream_velocity_inertial=freestream,
        body_panel_velocity_inertial=body_velocity_inertial,
        other_material_blob_velocity_inertial=other_blob_velocity,
        current_near_wake_velocity_inertial=near_wake_velocity,
        total_velocity_inertial=total,
        newborn_index=newborn_index,
    )


def march_attached_unsteady_outer_explicit_euler(
    *,
    surface: ActualSurface2D,
    kinematics: RigidKinematics2D,
    freestream_velocity_inertial: Any,
    time_step: float,
    history: AttachedOuterHistory2D,
    predicted_bound_circulation_change_ccw: float,
    newborn_core_radius: float,
    wall_transpiration_velocity: Any | None = None,
    initial_mean_emission_speed: float | None = None,
) -> AttachedOuterMarchResult2D:
    """Advance the complete attached S1 state by one owned Euler operation.

    The operation has one admissible order:

    ``history@t_n -> same-stage solve -> solved birth -> internally assembled
    material velocity -> explicit-Euler convection -> history@t_(n+1)``.

    No material velocity or replacement blob position is accepted from the
    caller.  This is the canonical multi-step path; the lower-level solve and
    immutable inventory helpers remain available for single-step equation
    audits.
    """
    if not isinstance(history, AttachedOuterHistory2D):
        raise SVIDWValidationError("history must be AttachedOuterHistory2D")
    inputs = AttachedOuterStepInput2D.from_history(
        surface=surface,
        kinematics=kinematics,
        freestream_velocity_inertial=freestream_velocity_inertial,
        time_step=time_step,
        history=history,
        predicted_bound_circulation_change_ccw=(predicted_bound_circulation_change_ccw),
        wall_transpiration_velocity=wall_transpiration_velocity,
        initial_mean_emission_speed=initial_mean_emission_speed,
    )
    solution = solve_attached_unsteady_outer_step(inputs)
    born = complete_attached_outer_history(
        solution,
        newborn_core_radius=newborn_core_radius,
    )
    advection = _material_advection_from_completed_step(solution, born)
    moved = convect_material_vortex_blobs(
        born.material_blobs,
        velocity_inertial=advection.total_velocity_inertial,
        time_step=inputs.time_step,
    )
    advanced = AttachedOuterHistory2D(
        bound_circulation_ccw=born.bound_circulation_ccw,
        material_blobs=moved,
        kelvin_reference_total_ccw=born.kelvin_reference_total_ccw,
        stage_time=born.stage_time + inputs.time_step,
    )
    return AttachedOuterMarchResult2D(
        solution=solution,
        history_at_stage_after_birth=born,
        advection=advection,
        history_next=advanced,
    )


def convect_attached_outer_history_explicit_euler(
    history: AttachedOuterHistory2D,
    *,
    velocity_inertial: Any,
    time_step: float,
) -> AttachedOuterHistory2D:
    """Reject the retired caller-velocity history path.

    The symbol remains as a fail-closed compatibility trap for early S1
    probes.  A typed history may now advance only through
    :func:`march_attached_unsteady_outer_explicit_euler`, which computes every
    material velocity from the same solved stage.
    """
    if not isinstance(history, AttachedOuterHistory2D):
        raise SVIDWValidationError("history must be AttachedOuterHistory2D")
    _array(
        "velocity_inertial",
        velocity_inertial,
        (len(history.material_blobs), 2),
    )
    _positive_scalar("time_step", time_step)
    raise SVIUnsteadyOuterScopeError(
        "caller-supplied material velocities are forbidden for history "
        "advancement; use march_attached_unsteady_outer_explicit_euler"
    )


def convect_material_vortex_blobs(
    blobs: Iterable[MaterialVortexBlob2D],
    *,
    velocity_inertial: Any,
    time_step: float,
) -> tuple[MaterialVortexBlob2D, ...]:
    """Advance old material blobs one explicit step without changing Gamma.

    The caller supplies the stage-consistent mean velocity at each blob.
    This separation keeps the material-conservation operation testable and
    avoids pretending S1 has selected a wake time integrator for N2.6e1.
    """
    items = tuple(blobs)
    if not all(isinstance(item, MaterialVortexBlob2D) for item in items):
        raise SVIDWValidationError("blobs must contain only MaterialVortexBlob2D")
    velocity = _array("velocity_inertial", velocity_inertial, (len(items), 2))
    delta_t = _positive_scalar("time_step", time_step)
    return tuple(
        MaterialVortexBlob2D(
            position_inertial=blob.position_inertial + delta_t * velocity[index],
            circulation_ccw=blob.circulation_ccw,
            core_radius=blob.core_radius,
        )
        for index, blob in enumerate(items)
    )
