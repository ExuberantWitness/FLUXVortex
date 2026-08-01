"""Read-only frozen-N1 field adapter for the actual-thickness shell.

This module belongs to diagnostic claim ``N3.1j3b6c``.  It does not solve
circulation, shed or convect a wake, compute pressure, or contribute force.
Its only aerodynamic operation is to replay the *already solved* N1
bound/wake velocity field at caller-supplied points with an explicit kernel,
arithmetic and symmetry identity.

The production half-wing identity is intentionally asymmetric:

* bound field = direct rings + orientation-correct image when ``sym=True``;
* wake field = direct pre-shed wake only.

A mirrored-wake field is returned only as a separately named physical
symmetry candidate.  It can never alter ``production_total``.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np

from .deforming_shell_kinematics import (
    DeformingDualSurfaceKinematics,
    deforming_dual_surface_kinematics,
)
from .hirato_shadow import mirrored_ring_field
from .thick_body_neumann_shadow import (
    RoboEagleClosedShell,
    close_roboeagle_dual_surface_shell,
)
from .viscous_shell_geometry import (
    DualSurfaceShell,
    naca4_half_thickness,
)


class N1ThickShellAdapterError(ValueError):
    """Invalid N1 snapshot, field convention, or shell mapping."""


class N1SymmetryRole(str, Enum):
    PRODUCTION_IDENTITY = "production_identity"
    PHYSICAL_SYMMETRY_CANDIDATE = "physical_symmetry_candidate"


def _finite_copy(
    name: str,
    value: Any,
    *,
    shape: tuple[int, ...] | None = None,
    ndim: int | None = None,
) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if shape is not None and array.shape != shape:
        raise N1ThickShellAdapterError(
            f"{name} must have shape {shape}, got {array.shape}"
        )
    if ndim is not None and array.ndim != ndim:
        raise N1ThickShellAdapterError(
            f"{name} must have ndim={ndim}, got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise N1ThickShellAdapterError(f"{name} contains non-finite values")
    result = array.copy()
    result.flags.writeable = False
    return result


def _required_text(frame: Mapping[str, Any], key: str) -> str:
    if key not in frame or not isinstance(frame[key], str) or not frame[key]:
        raise N1ThickShellAdapterError(f"snapshot requires non-empty {key}")
    return frame[key]


@dataclass(frozen=True)
class N1StepSnapshot:
    time: float
    nc: int
    ns: int
    bound_rings: np.ndarray
    bound_gamma: np.ndarray
    mean_surface: np.ndarray
    mean_surface_velocity: np.ndarray
    collocation_points: np.ndarray
    panel_normals: np.ndarray
    collocation_wall_velocity: np.ndarray
    captured_wake_velocity: np.ndarray
    wake_rings: np.ndarray
    wake_gamma: np.ndarray
    wake_type: np.ndarray
    freestream_velocity: np.ndarray
    symmetry_enabled: bool
    snapshot_phase: str
    bound_kernel_kind: str
    bound_arithmetic: str
    bound_denominator_floor: float
    wake_kernel_kind: str
    wake_arithmetic: str
    wake_denominator_floor: float
    wake_core_delta: np.ndarray

    @classmethod
    def from_frame(cls, frame: Mapping[str, Any]) -> "N1StepSnapshot":
        if not isinstance(frame, Mapping):
            raise N1ThickShellAdapterError("snapshot must be a mapping")
        try:
            nc = int(frame["nc"])
            ns = int(frame["ns"])
            time = float(frame["t"])
        except (KeyError, TypeError, ValueError) as error:
            raise N1ThickShellAdapterError(
                "snapshot requires finite t and integer nc/ns"
            ) from error
        if (
            nc <= 0
            or ns <= 0
            or not np.isfinite(time)
            or int(frame["nc"]) != frame["nc"]
            or int(frame["ns"]) != frame["ns"]
        ):
            raise N1ThickShellAdapterError("invalid t, nc, or ns")
        panel_count = nc * ns
        node_shape = (nc + 1, ns + 1, 3)
        wake_rings = _finite_copy("wr", frame.get("wr"), ndim=3)
        if wake_rings.shape[1:] != (4, 3):
            raise N1ThickShellAdapterError(
                f"wr must have shape (nw,4,3), got {wake_rings.shape}"
            )
        wake_count = len(wake_rings)
        wake_type_raw = np.asarray(frame.get("wtype"))
        if (
            wake_type_raw.shape != (wake_count,)
            or not np.issubdtype(wake_type_raw.dtype, np.integer)
        ):
            raise N1ThickShellAdapterError(
                "wtype must contain one integer identity per wake ring"
            )
        wake_type = wake_type_raw.astype(int, copy=True)
        wake_type.flags.writeable = False
        bound_kind = _required_text(frame, "bound_kernel_kind")
        wake_kind = _required_text(frame, "wake_kernel_kind")
        bound_arithmetic = _required_text(frame, "bound_arithmetic")
        wake_arithmetic = _required_text(frame, "wake_arithmetic")
        if bound_kind != "singular" or bound_arithmetic != "fp64":
            raise N1ThickShellAdapterError(
                "frozen N1 bound field requires singular/fp64 identity"
            )
        if wake_kind not in {"singular", "van_garrel_core"}:
            raise N1ThickShellAdapterError(
                f"unknown wake kernel kind {wake_kind!r}"
            )
        if wake_arithmetic not in {"fp64", "fp32"}:
            raise N1ThickShellAdapterError(
                f"unknown wake arithmetic {wake_arithmetic!r}"
            )
        bound_floor = float(frame.get("bound_denominator_floor"))
        wake_floor = float(frame.get("wake_denominator_floor"))
        if (
            not np.isfinite(bound_floor)
            or bound_floor <= 0.0
            or not np.isfinite(wake_floor)
            or wake_floor <= 0.0
        ):
            raise N1ThickShellAdapterError(
                "kernel denominator floors must be positive and finite"
            )
        core = _finite_copy(
            "wake_core_delta",
            frame.get("wake_core_delta"),
            shape=(wake_count,),
        )
        if np.any(core < 0.0):
            raise N1ThickShellAdapterError(
                "wake_core_delta cannot be negative"
            )
        if wake_kind == "singular" and np.any(core != 0.0):
            raise N1ThickShellAdapterError(
                "singular wake identity cannot carry a core"
            )
        if (
            wake_kind == "van_garrel_core"
            and wake_count > 0
            and np.any(core <= 0.0)
        ):
            raise N1ThickShellAdapterError(
                "cored wake identity requires every per-ring core"
            )
        phase = _required_text(frame, "snapshot_phase")
        if phase != "post_force_pre_shed":
            raise N1ThickShellAdapterError(
                f"unsupported snapshot phase {phase!r}"
            )
        return cls(
            time=time,
            nc=nc,
            ns=ns,
            bound_rings=_finite_copy(
                "bound", frame.get("bound"), shape=(panel_count, 4, 3)
            ),
            bound_gamma=_finite_copy(
                "gam", frame.get("gam"), shape=(panel_count,)
            ),
            mean_surface=_finite_copy(
                "corners", frame.get("corners"), shape=node_shape
            ),
            mean_surface_velocity=_finite_copy(
                "corner_velocity",
                frame.get("corner_velocity"),
                shape=node_shape,
            ),
            collocation_points=_finite_copy(
                "collocation_points",
                frame.get("collocation_points"),
                shape=(panel_count, 3),
            ),
            panel_normals=_finite_copy(
                "panel_normals",
                frame.get("panel_normals"),
                shape=(panel_count, 3),
            ),
            collocation_wall_velocity=_finite_copy(
                "collocation_velocity",
                frame.get("collocation_velocity"),
                shape=(panel_count, 3),
            ),
            captured_wake_velocity=_finite_copy(
                "wake_induced_velocity",
                frame.get("wake_induced_velocity"),
                shape=(panel_count, 3),
            ),
            wake_rings=wake_rings,
            wake_gamma=_finite_copy(
                "wg", frame.get("wg"), shape=(wake_count,)
            ),
            wake_type=wake_type,
            freestream_velocity=_finite_copy(
                "freestream_velocity",
                frame.get("freestream_velocity"),
                shape=(3,),
            ),
            symmetry_enabled=bool(frame.get("symmetry_enabled")),
            snapshot_phase=phase,
            bound_kernel_kind=bound_kind,
            bound_arithmetic=bound_arithmetic,
            bound_denominator_floor=bound_floor,
            wake_kernel_kind=wake_kind,
            wake_arithmetic=wake_arithmetic,
            wake_denominator_floor=wake_floor,
            wake_core_delta=core,
        )


@dataclass(frozen=True)
class N1SnapshotTriplet:
    previous: N1StepSnapshot
    current: N1StepSnapshot
    following: N1StepSnapshot
    dt: float


@dataclass(frozen=True)
class N1IncidentVelocityLedger:
    freestream: np.ndarray
    bound_direct: np.ndarray
    bound_image: np.ndarray
    wake_direct: np.ndarray
    wake_image_candidate: np.ndarray
    production_total: np.ndarray
    physical_symmetry_candidate_total: np.ndarray

    def selected(self, role: N1SymmetryRole | str) -> np.ndarray:
        try:
            identity = N1SymmetryRole(role)
        except ValueError as error:
            raise N1ThickShellAdapterError(
                f"unknown symmetry role {role!r}"
            ) from error
        if identity is N1SymmetryRole.PRODUCTION_IDENTITY:
            return self.production_total.copy()
        return self.physical_symmetry_candidate_total.copy()


@dataclass(frozen=True)
class N1CollocationReplay:
    ledger: N1IncidentVelocityLedger
    normal_residual: np.ndarray
    maximum_normal_residual: float
    relative_normal_residual: float
    captured_wake_velocity_error: np.ndarray
    maximum_captured_wake_velocity_error: float
    relative_captured_wake_velocity_error: float


@dataclass(frozen=True)
class N1ActualShellKinematics:
    material_chord_fraction: np.ndarray
    material_span_coordinate: np.ndarray
    kinematics: DeformingDualSurfaceKinematics
    dual_surface_shell: DualSurfaceShell
    closed_shell: RoboEagleClosedShell
    vertex_wall_velocity: np.ndarray
    face_wall_velocity: np.ndarray
    maximum_mean_surface_change: float
    maximum_material_pairing_error: float
    maximum_wall_velocity_node_mapping_error: float


def parse_n1_snapshot_triplet(
    frames: Sequence[Mapping[str, Any] | N1StepSnapshot],
    *,
    expected_dt: float | None = None,
) -> N1SnapshotTriplet:
    if len(frames) != 3:
        raise N1ThickShellAdapterError(
            "exactly three consecutive snapshots are required"
        )
    parsed = tuple(
        item if isinstance(item, N1StepSnapshot)
        else N1StepSnapshot.from_frame(item)
        for item in frames
    )
    previous, current, following = parsed
    identity = (
        current.nc,
        current.ns,
        current.symmetry_enabled,
        current.bound_kernel_kind,
        current.bound_arithmetic,
        current.wake_kernel_kind,
        current.wake_arithmetic,
    )
    for snapshot in (previous, following):
        other = (
            snapshot.nc,
            snapshot.ns,
            snapshot.symmetry_enabled,
            snapshot.bound_kernel_kind,
            snapshot.bound_arithmetic,
            snapshot.wake_kernel_kind,
            snapshot.wake_arithmetic,
        )
        if other != identity:
            raise N1ThickShellAdapterError(
                "snapshot triplet changes its grid or field convention"
            )
    dt_before = current.time - previous.time
    dt_after = following.time - current.time
    scale = max(abs(current.time), abs(dt_before), abs(dt_after), 1.0)
    tolerance = 64.0 * np.finfo(float).eps * scale
    if (
        dt_before <= 0.0
        or dt_after <= 0.0
        or abs(dt_before - dt_after) > tolerance
    ):
        raise N1ThickShellAdapterError(
            "snapshot triplet must be consecutive and equally spaced"
        )
    if expected_dt is not None:
        expected = float(expected_dt)
        if (
            not np.isfinite(expected)
            or expected <= 0.0
            or abs(dt_before - expected) > tolerance
        ):
            raise N1ThickShellAdapterError(
                f"snapshot spacing {dt_before} differs from expected {expected}"
            )
    return N1SnapshotTriplet(
        previous=previous,
        current=current,
        following=following,
        dt=dt_before,
    )


def production_ring_field_velocity(
    points: Any,
    rings: Any,
    gamma: Any,
    *,
    arithmetic: str,
    denominator_floor: float,
    core_delta: Any | None = None,
) -> np.ndarray:
    targets = _finite_copy("points", points, ndim=2)
    geometry = _finite_copy("rings", rings, ndim=3)
    strength = _finite_copy("gamma", gamma, ndim=1)
    if targets.shape[1] != 3 or geometry.shape[1:] != (4, 3):
        raise N1ThickShellAdapterError(
            "points/rings require shapes (m,3) and (n,4,3)"
        )
    if strength.shape != (len(geometry),):
        raise N1ThickShellAdapterError(
            "gamma must contain one value per ring"
        )
    if (
        arithmetic not in {"fp64", "fp32"}
        or not np.isfinite(denominator_floor)
        or denominator_floor <= 0.0
    ):
        raise N1ThickShellAdapterError(
            "unknown arithmetic or invalid denominator floor"
        )
    if core_delta is None:
        core = np.zeros(len(geometry), dtype=float)
    else:
        core = _finite_copy(
            "core_delta", core_delta, shape=(len(geometry),)
        )
    if np.any(core < 0.0):
        raise N1ThickShellAdapterError("core_delta cannot be negative")

    dtype = np.float64 if arithmetic == "fp64" else np.float32
    target_d = np.asarray(targets, dtype=dtype)
    velocity = np.zeros(target_d.shape, dtype=dtype)
    pi_factor = dtype(0.07957747154594767)
    norm_floor = dtype(1.0e-20 if arithmetic == "fp64" else 1.0e-12)
    denominator = dtype(denominator_floor)
    for ring, circulation, delta in zip(
        geometry, strength, core, strict=True
    ):
        ring_d = np.asarray(ring, dtype=dtype)
        circulation_d = dtype(circulation)
        delta_d = dtype(delta)
        for start, end in ((0, 1), (1, 2), (2, 3), (3, 0)):
            r1 = target_d - ring_d[start]
            r2 = target_d - ring_d[end]
            r0 = ring_d[end] - ring_d[start]
            cross = np.cross(r1, r2)
            cross_square = np.sum(cross * cross, axis=1, dtype=dtype)
            segment_square = np.sum(r0 * r0, dtype=dtype)
            divisor = (
                cross_square + delta_d * delta_d * segment_square
                + denominator
            )
            norm1 = np.sqrt(
                np.sum(r1 * r1, axis=1, dtype=dtype) + norm_floor
            )
            norm2 = np.sqrt(
                np.sum(r2 * r2, axis=1, dtype=dtype) + norm_floor
            )
            direction = r1 / norm1[:, None] - r2 / norm2[:, None]
            coefficient = np.sum(direction * r0, axis=1, dtype=dtype)
            velocity = velocity + (
                circulation_d
                * pi_factor
                * coefficient[:, None]
                / divisor[:, None]
                * cross
            )
    return np.asarray(velocity, dtype=float)


def evaluate_n1_incident_velocity(
    snapshot: N1StepSnapshot,
    points: Any,
) -> N1IncidentVelocityLedger:
    if not isinstance(snapshot, N1StepSnapshot):
        raise N1ThickShellAdapterError(
            "snapshot must be an N1StepSnapshot"
        )
    targets = _finite_copy("points", points, ndim=2)
    if targets.shape[1] != 3:
        raise N1ThickShellAdapterError("points must have shape (m,3)")
    bound_direct = production_ring_field_velocity(
        targets,
        snapshot.bound_rings,
        snapshot.bound_gamma,
        arithmetic=snapshot.bound_arithmetic,
        denominator_floor=snapshot.bound_denominator_floor,
    )
    if snapshot.symmetry_enabled:
        bound_image = production_ring_field_velocity(
            targets,
            mirrored_ring_field(snapshot.bound_rings),
            snapshot.bound_gamma,
            arithmetic=snapshot.bound_arithmetic,
            denominator_floor=snapshot.bound_denominator_floor,
        )
    else:
        bound_image = np.zeros_like(bound_direct)
    wake_core = (
        snapshot.wake_core_delta
        if snapshot.wake_kernel_kind == "van_garrel_core"
        else None
    )
    wake_direct = production_ring_field_velocity(
        targets,
        snapshot.wake_rings,
        snapshot.wake_gamma,
        arithmetic=snapshot.wake_arithmetic,
        denominator_floor=snapshot.wake_denominator_floor,
        core_delta=wake_core,
    )
    if snapshot.symmetry_enabled:
        wake_image = production_ring_field_velocity(
            targets,
            mirrored_ring_field(snapshot.wake_rings),
            snapshot.wake_gamma,
            arithmetic=snapshot.wake_arithmetic,
            denominator_floor=snapshot.wake_denominator_floor,
            core_delta=wake_core,
        )
    else:
        wake_image = np.zeros_like(wake_direct)
    freestream = np.broadcast_to(
        snapshot.freestream_velocity, targets.shape
    ).copy()
    production = freestream + bound_direct + bound_image + wake_direct
    candidate = production + wake_image
    return N1IncidentVelocityLedger(
        freestream=freestream,
        bound_direct=bound_direct,
        bound_image=bound_image,
        wake_direct=wake_direct,
        wake_image_candidate=wake_image,
        production_total=production,
        physical_symmetry_candidate_total=candidate,
    )


def replay_n1_collocation_boundary(
    snapshot: N1StepSnapshot,
) -> N1CollocationReplay:
    ledger = evaluate_n1_incident_velocity(
        snapshot, snapshot.collocation_points
    )
    relative_velocity = (
        ledger.production_total - snapshot.collocation_wall_velocity
    )
    residual = np.einsum(
        "ij,ij->i", relative_velocity, snapshot.panel_normals
    )
    velocity_scale = max(
        float(np.max(np.linalg.norm(relative_velocity, axis=1), initial=0.0)),
        float(np.linalg.norm(snapshot.freestream_velocity)),
        np.finfo(float).tiny,
    )
    wake_error = (
        ledger.wake_direct - snapshot.captured_wake_velocity
    )
    wake_scale = max(
        float(
            np.max(
                np.linalg.norm(snapshot.captured_wake_velocity, axis=1),
                initial=0.0,
            )
        ),
        float(np.linalg.norm(snapshot.freestream_velocity)),
        np.finfo(float).tiny,
    )
    return N1CollocationReplay(
        ledger=ledger,
        normal_residual=residual,
        maximum_normal_residual=float(
            np.max(np.abs(residual), initial=0.0)
        ),
        relative_normal_residual=float(
            np.max(np.abs(residual), initial=0.0) / velocity_scale
        ),
        captured_wake_velocity_error=wake_error,
        maximum_captured_wake_velocity_error=float(
            np.max(np.linalg.norm(wake_error, axis=1), initial=0.0)
        ),
        relative_captured_wake_velocity_error=float(
            np.max(np.linalg.norm(wake_error, axis=1), initial=0.0)
            / wake_scale
        ),
    )


def _material_coordinates(
    mean_surface: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    chord_vector = mean_surface[-1] - mean_surface[0]
    chord_square = np.einsum("ij,ij->i", chord_vector, chord_vector)
    if np.any(chord_square <= 0.0):
        raise N1ThickShellAdapterError(
            "mean surface contains a zero-length section chord"
        )
    relative = mean_surface - mean_surface[0][None, :, :]
    xi_by_span = np.einsum(
        "ijk,jk->ij", relative, chord_vector
    ) / chord_square[None, :]
    xi = np.mean(xi_by_span, axis=1)
    scale = max(float(np.max(np.sqrt(chord_square))), 1.0)
    xi_error = float(
        np.max(np.abs(xi_by_span - xi[:, None]), initial=0.0)
    )
    if xi_error > 2.0e-11 * scale or np.any(np.diff(xi) <= 0.0):
        raise N1ThickShellAdapterError(
            f"cannot recover one shared chord coordinate; error={xi_error}"
        )
    xi[0] = 0.0
    xi[-1] = 1.0
    leading_edge = mean_surface[0]
    eta = np.zeros(len(leading_edge), dtype=float)
    eta[1:] = np.cumsum(
        np.linalg.norm(np.diff(leading_edge, axis=0), axis=1)
    )
    if np.any(np.diff(eta) <= 0.0):
        raise N1ThickShellAdapterError(
            "cannot recover a strictly increasing material span coordinate"
        )
    return xi, eta, np.sqrt(chord_square)


def build_n1_actual_shell_kinematics(
    triplet: N1SnapshotTriplet,
    *,
    thickness_ratio: float = 0.06,
) -> N1ActualShellKinematics:
    if not isinstance(triplet, N1SnapshotTriplet):
        raise N1ThickShellAdapterError(
            "triplet must be an N1SnapshotTriplet"
        )
    return build_actual_shell_kinematics_from_mean_surfaces(
        previous_mean_surface=triplet.previous.mean_surface,
        current_mean_surface=triplet.current.mean_surface,
        next_mean_surface=triplet.following.mean_surface,
        dt=triplet.dt,
        thickness_ratio=thickness_ratio,
    )


def build_actual_shell_kinematics_from_mean_surfaces(
    *,
    previous_mean_surface: Any,
    current_mean_surface: Any,
    next_mean_surface: Any,
    dt: float,
    thickness_ratio: float = 0.06,
) -> N1ActualShellKinematics:
    """Build a diagnostic shell on a grid independent of the N1 lattice.

    The supplied surfaces must use the same material topology and three
    consecutive times.  This overload permits shell-refinement studies while
    the frozen N1 bound/wake state remains unchanged.
    """
    previous = _finite_copy(
        "previous_mean_surface", previous_mean_surface, ndim=3
    )
    current_mean = _finite_copy(
        "current_mean_surface", current_mean_surface, ndim=3
    )
    following = _finite_copy(
        "next_mean_surface", next_mean_surface, ndim=3
    )
    if (
        previous.shape != current_mean.shape
        or following.shape != current_mean.shape
        or current_mean.shape[2] != 3
        or current_mean.shape[0] < 3
        or current_mean.shape[1] < 3
    ):
        raise N1ThickShellAdapterError(
            "three mean surfaces must share shape (nxi>=3,neta>=3,3)"
        )
    spacing = float(dt)
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise N1ThickShellAdapterError("dt must be positive and finite")
    xi, eta, chord = _material_coordinates(current_mean)
    half_thickness = (
        naca4_half_thickness(
            xi,
            thickness_ratio=thickness_ratio,
            closed_trailing_edge=False,
        )[:, None]
        * chord[None, :]
    )
    kinematics = deforming_dual_surface_kinematics(
        previous_mean_surface=previous,
        current_mean_surface=current_mean,
        next_mean_surface=following,
        xi=xi,
        eta=eta,
        half_thickness=half_thickness,
        dt=spacing,
    )
    current = kinematics.current
    shell = DualSurfaceShell(
        chord_fraction=xi.copy(),
        span=eta.copy(),
        chord=chord.copy(),
        mean_surface=current.mean_surface.copy(),
        upper_surface=current.upper_surface.copy(),
        lower_surface=current.lower_surface.copy(),
        half_thickness=current.half_thickness.copy(),
        section_director=current.director.copy(),
    )
    closed = close_roboeagle_dual_surface_shell(shell)
    vertex_velocity = np.zeros_like(closed.mesh.vertices)
    velocity_count = np.zeros(len(vertex_velocity), dtype=int)
    for node_velocity, indices in (
        (kinematics.upper_velocity, closed.upper_vertex_indices),
        (kinematics.lower_velocity, closed.lower_vertex_indices),
    ):
        for material_index in np.ndindex(indices.shape):
            vertex_index = int(indices[material_index])
            vertex_velocity[vertex_index] += node_velocity[material_index]
            velocity_count[vertex_index] += 1
    if np.any(velocity_count == 0):
        raise N1ThickShellAdapterError(
            "closed shell contains an unmapped wall vertex"
        )
    vertex_velocity /= velocity_count[:, None]
    mapping_error = 0.0
    for node_velocity, indices in (
        (kinematics.upper_velocity, closed.upper_vertex_indices),
        (kinematics.lower_velocity, closed.lower_vertex_indices),
    ):
        mapped = vertex_velocity[indices]
        mapping_error = max(
            mapping_error,
            float(np.max(np.abs(mapped - node_velocity), initial=0.0)),
        )
    face_velocity = np.mean(
        vertex_velocity[closed.mesh.faces], axis=1
    )
    mean_change = float(
        np.max(
            np.abs(current.mean_surface - current_mean),
            initial=0.0,
        )
    )
    pairing_error = float(
        np.max(
            np.abs(
                0.5 * (current.upper_surface + current.lower_surface)
                - current.mean_surface
            ),
            initial=0.0,
        )
    )
    return N1ActualShellKinematics(
        material_chord_fraction=xi,
        material_span_coordinate=eta,
        kinematics=kinematics,
        dual_surface_shell=shell,
        closed_shell=closed,
        vertex_wall_velocity=vertex_velocity,
        face_wall_velocity=face_velocity,
        maximum_mean_surface_change=mean_change,
        maximum_material_pairing_error=pairing_error,
        maximum_wall_velocity_node_mapping_error=mapping_error,
    )
