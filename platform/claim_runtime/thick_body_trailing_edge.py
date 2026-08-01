"""Read-only trailing-edge residuals for an actual-thickness N1 shadow.

The residuals deliberately do not choose a Kutta closure.  They keep the
Kemp B-1/B-2 direction alternatives, closed-shell source compatibility and
the thin-lattice wake topology in separate channels.  A pressure residual is
not fabricated here because it requires a qualified total material-potential
history and a unified Bernoulli evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .hirato_shadow import mirrored_ring_field
from .n1_thick_shell_adapter import (
    N1ActualShellKinematics,
    N1IncidentVelocityLedger,
    N1StepSnapshot,
    N1ThickShellAdapterError,
)
from .thick_body_neumann_shadow import (
    ClosedTriangularMesh,
    NeumannSourceSolution,
)


@dataclass(frozen=True)
class TrailingEdgeTopologyResidual:
    base_thickness: np.ndarray
    base_thickness_over_chord: np.ndarray
    n1_rear_line_offset: np.ndarray
    n1_rear_line_offset_over_chord: np.ndarray
    rear_line_kernel_identity_error: float
    base_to_rear_offset_ratio: np.ndarray
    first_wake_step_over_chord: np.ndarray
    actual_topology: str
    frozen_n1_topology: str


@dataclass(frozen=True)
class TrailingEdgeDirectionResidual:
    upper_face_indices: np.ndarray
    lower_face_indices: np.ndarray
    upper_tangent_normal_velocity: np.ndarray
    lower_tangent_normal_velocity: np.ndarray
    bisector_normal_velocity: np.ndarray
    upper_relative_speed: np.ndarray
    lower_relative_speed: np.ndarray
    source_flux: float
    relative_source_flux: float
    pressure_residual_available: bool
    pressure_residual_blocker: str


@dataclass(frozen=True)
class ClosedSurfaceFluxLedger:
    freestream: float
    bound_direct: float
    bound_image: float
    wake_direct: float
    wake_image_candidate: float
    production_total: float
    physical_symmetry_candidate_total: float
    wall_volume_flux: float
    production_reconstruction_error: float
    physical_symmetry_reconstruction_error: float


@dataclass(frozen=True)
class ActiveFilament:
    start: np.ndarray
    end: np.ndarray
    circulation: float
    endpoint_classes: tuple[str, str]
    proper_intersection_count: int
    intersection_face_roles: tuple[str, ...]
    minimum_shell_distance: float


@dataclass(frozen=True)
class FilamentChannelTopology:
    channel: str
    raw_segment_count: int
    unique_segment_count: int
    active_unique_segment_count: int
    cancelled_unique_segment_count: int
    inside_inside_segment_count: int
    outside_outside_segment_count: int
    inside_outside_segment_count: int
    boundary_endpoint_segment_count: int
    shell_contact_segment_count: int
    proper_shell_piercing_segment_count: int
    proper_shell_intersection_count: int
    proper_intersection_face_role_counts: tuple[tuple[str, int], ...]
    minimum_active_segment_to_shell_distance: float | None
    circulation_weighted_piercing_fraction: float
    topology_tolerance: float
    circulation_tolerance: float
    piercing_examples: tuple[ActiveFilament, ...]


@dataclass(frozen=True)
class N1FilamentShellTopologyAudit:
    bound_direct: FilamentChannelTopology
    bound_image: FilamentChannelTopology
    wake_direct: FilamentChannelTopology
    wake_image_candidate: FilamentChannelTopology


def _unit(name: str, vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms <= 0.0) or not np.all(np.isfinite(norms)):
        raise N1ThickShellAdapterError(
            f"{name} contains a zero or non-finite vector"
        )
    return vectors / norms[:, None]


def closed_surface_flux_ledger(
    mesh: ClosedTriangularMesh,
    incident: N1IncidentVelocityLedger,
    *,
    wall_velocity: Any,
) -> ClosedSurfaceFluxLedger:
    """Integrate each frozen-N1 velocity channel over one closed shell."""
    if not isinstance(mesh, ClosedTriangularMesh):
        raise N1ThickShellAdapterError(
            "mesh must be a ClosedTriangularMesh"
        )
    if not isinstance(incident, N1IncidentVelocityLedger):
        raise N1ThickShellAdapterError(
            "incident must be an N1IncidentVelocityLedger"
        )
    wall = np.asarray(wall_velocity, dtype=float)
    expected = (len(mesh.faces), 3)
    channels = {
        "freestream": incident.freestream,
        "bound_direct": incident.bound_direct,
        "bound_image": incident.bound_image,
        "wake_direct": incident.wake_direct,
        "wake_image_candidate": incident.wake_image_candidate,
        "production_total": incident.production_total,
        "physical_symmetry_candidate_total":
            incident.physical_symmetry_candidate_total,
    }
    if wall.shape != expected or not np.all(np.isfinite(wall)):
        raise N1ThickShellAdapterError(
            f"wall_velocity must have shape {expected}"
        )
    for name, velocity in channels.items():
        if velocity.shape != expected or not np.all(np.isfinite(velocity)):
            raise N1ThickShellAdapterError(
                f"{name} must have shape {expected} and finite values"
            )
    area_normal = mesh.areas[:, None] * mesh.normals

    def flux(velocity: np.ndarray) -> float:
        return float(np.sum(velocity * area_normal))

    values = {name: flux(value) for name, value in channels.items()}
    production_sum = (
        values["freestream"]
        + values["bound_direct"]
        + values["bound_image"]
        + values["wake_direct"]
    )
    candidate_sum = (
        production_sum + values["wake_image_candidate"]
    )
    return ClosedSurfaceFluxLedger(
        freestream=values["freestream"],
        bound_direct=values["bound_direct"],
        bound_image=values["bound_image"],
        wake_direct=values["wake_direct"],
        wake_image_candidate=values["wake_image_candidate"],
        production_total=values["production_total"],
        physical_symmetry_candidate_total=values[
            "physical_symmetry_candidate_total"
        ],
        wall_volume_flux=flux(wall),
        production_reconstruction_error=(
            values["production_total"] - production_sum
        ),
        physical_symmetry_reconstruction_error=(
            values["physical_symmetry_candidate_total"] - candidate_sum
        ),
    )


def _closest_points_on_triangles(
    point: np.ndarray,
    triangles: np.ndarray,
) -> np.ndarray:
    """Ericson point-triangle closest points, vectorized over triangles."""
    a = triangles[:, 0]
    b = triangles[:, 1]
    c = triangles[:, 2]
    ab = b - a
    ac = c - a
    ap = point - a
    d1 = np.einsum("ij,ij->i", ab, ap)
    d2 = np.einsum("ij,ij->i", ac, ap)
    closest = np.empty_like(a)
    assigned = np.zeros(len(triangles), dtype=bool)

    mask = (d1 <= 0.0) & (d2 <= 0.0)
    closest[mask] = a[mask]
    assigned |= mask

    bp = point - b
    d3 = np.einsum("ij,ij->i", ab, bp)
    d4 = np.einsum("ij,ij->i", ac, bp)
    mask = (~assigned) & (d3 >= 0.0) & (d4 <= d3)
    closest[mask] = b[mask]
    assigned |= mask

    vc = d1 * d4 - d3 * d2
    mask = (
        (~assigned)
        & (vc <= 0.0)
        & (d1 >= 0.0)
        & (d3 <= 0.0)
    )
    denominator = d1 - d3
    fraction = np.divide(
        d1,
        denominator,
        out=np.zeros_like(d1),
        where=denominator != 0.0,
    )
    closest[mask] = a[mask] + fraction[mask, None] * ab[mask]
    assigned |= mask

    cp = point - c
    d5 = np.einsum("ij,ij->i", ab, cp)
    d6 = np.einsum("ij,ij->i", ac, cp)
    mask = (~assigned) & (d6 >= 0.0) & (d5 <= d6)
    closest[mask] = c[mask]
    assigned |= mask

    vb = d5 * d2 - d1 * d6
    mask = (
        (~assigned)
        & (vb <= 0.0)
        & (d2 >= 0.0)
        & (d6 <= 0.0)
    )
    denominator = d2 - d6
    fraction = np.divide(
        d2,
        denominator,
        out=np.zeros_like(d2),
        where=denominator != 0.0,
    )
    closest[mask] = a[mask] + fraction[mask, None] * ac[mask]
    assigned |= mask

    va = d3 * d6 - d5 * d4
    mask = (
        (~assigned)
        & (va <= 0.0)
        & ((d4 - d3) >= 0.0)
        & ((d5 - d6) >= 0.0)
    )
    denominator = (d4 - d3) + (d5 - d6)
    fraction = np.divide(
        d4 - d3,
        denominator,
        out=np.zeros_like(d3),
        where=denominator != 0.0,
    )
    closest[mask] = b[mask] + fraction[mask, None] * (c - b)[mask]
    assigned |= mask

    remaining = ~assigned
    denominator = va + vb + vc
    v = np.divide(
        vb,
        denominator,
        out=np.zeros_like(vb),
        where=denominator != 0.0,
    )
    w = np.divide(
        vc,
        denominator,
        out=np.zeros_like(vc),
        where=denominator != 0.0,
    )
    closest[remaining] = (
        a[remaining]
        + v[remaining, None] * ab[remaining]
        + w[remaining, None] * ac[remaining]
    )
    return closest


def _point_triangle_distance(
    point: np.ndarray,
    triangles: np.ndarray,
) -> float:
    closest = _closest_points_on_triangles(point, triangles)
    return float(
        np.min(np.linalg.norm(closest - point, axis=1), initial=np.inf)
    )


def _point_to_segments_distance(
    points: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> np.ndarray:
    direction = end - start
    square = np.einsum("ij,ij->i", direction, direction)
    parameter = np.divide(
        np.einsum("ij,ij->i", points - start, direction),
        square,
        out=np.zeros_like(square),
        where=square > 0.0,
    )
    parameter = np.clip(parameter, 0.0, 1.0)
    closest = start + parameter[:, None] * direction
    return np.linalg.norm(points - closest, axis=1)


def _segment_segment_distance_to_many(
    start: np.ndarray,
    end: np.ndarray,
    other_start: np.ndarray,
    other_end: np.ndarray,
) -> np.ndarray:
    """Exact minimum distance from one segment to many segments."""
    direction = end - start
    other_direction = other_end - other_start
    a = float(np.dot(direction, direction))
    c = np.einsum("ij,ij->i", other_direction, other_direction)
    if a <= 0.0 or np.any(c <= 0.0):
        raise N1ThickShellAdapterError(
            "filament or mesh edge contains a zero-length segment"
        )
    b = other_direction @ direction
    relative = start - other_start
    d = relative @ direction
    e = np.einsum("ij,ij->i", other_direction, relative)
    denominator = a * c - b * b
    s = np.divide(
        b * e - c * d,
        denominator,
        out=np.zeros_like(denominator),
        where=np.abs(denominator) > np.finfo(float).tiny,
    )
    t = np.divide(
        a * e - b * d,
        denominator,
        out=np.zeros_like(denominator),
        where=np.abs(denominator) > np.finfo(float).tiny,
    )
    interior = (
        (s >= 0.0) & (s <= 1.0) & (t >= 0.0) & (t <= 1.0)
    )
    interior_delta = (
        relative + s[:, None] * direction
        - t[:, None] * other_direction
    )
    distances = np.full(len(other_start), np.inf)
    distances[interior] = np.linalg.norm(
        interior_delta[interior], axis=1
    )
    point_start = np.broadcast_to(start, other_start.shape)
    point_end = np.broadcast_to(end, other_start.shape)
    distances = np.minimum(
        distances,
        _point_to_segments_distance(
            point_start, other_start, other_end
        ),
    )
    distances = np.minimum(
        distances,
        _point_to_segments_distance(
            point_end, other_start, other_end
        ),
    )
    filament_start = np.broadcast_to(start, other_start.shape)
    filament_end = np.broadcast_to(end, other_end.shape)
    distances = np.minimum(
        distances,
        _point_to_segments_distance(
            other_start, filament_start, filament_end
        ),
    )
    distances = np.minimum(
        distances,
        _point_to_segments_distance(
            other_end, filament_start, filament_end
        ),
    )
    return distances


def _proper_segment_triangle_hits(
    start: np.ndarray,
    end: np.ndarray,
    triangles: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
    direction = end - start
    length = float(np.linalg.norm(direction))
    edge1 = triangles[:, 1] - triangles[:, 0]
    edge2 = triangles[:, 2] - triangles[:, 0]
    h = np.cross(np.broadcast_to(direction, edge2.shape), edge2)
    determinant = np.einsum("ij,ij->i", edge1, h)
    determinant_floor = tolerance * np.maximum(
        np.linalg.norm(edge1, axis=1)
        * np.linalg.norm(edge2, axis=1)
        * max(length, tolerance),
        np.finfo(float).tiny,
    )
    valid = np.abs(determinant) > determinant_floor
    inverse = np.divide(
        1.0,
        determinant,
        out=np.zeros_like(determinant),
        where=valid,
    )
    s = start - triangles[:, 0]
    u = inverse * np.einsum("ij,ij->i", s, h)
    q = np.cross(s, edge1)
    v = inverse * (q @ direction)
    parameter = inverse * np.einsum("ij,ij->i", edge2, q)
    parameter_tolerance = tolerance / max(length, tolerance)
    hit = (
        valid
        & (u >= -parameter_tolerance)
        & (v >= -parameter_tolerance)
        & ((u + v) <= 1.0 + parameter_tolerance)
        & (parameter > parameter_tolerance)
        & (parameter < 1.0 - parameter_tolerance)
    )
    values = np.sort(parameter[hit])
    if len(values) == 0:
        return values, ()
    hit_indices = np.flatnonzero(hit)
    order = np.argsort(parameter[hit])
    values = parameter[hit][order]
    hit_indices = hit_indices[order]
    unique = [float(values[0])]
    groups: list[list[int]] = [[int(hit_indices[0])]]
    for value, face_index in zip(
        values[1:], hit_indices[1:], strict=True
    ):
        if abs(float(value) - unique[-1]) > 4.0 * parameter_tolerance:
            unique.append(float(value))
            groups.append([int(face_index)])
        else:
            groups[-1].append(int(face_index))
    return np.asarray(unique), tuple(
        tuple(group) for group in groups
    )


def _proper_segment_triangle_parameters(
    start: np.ndarray,
    end: np.ndarray,
    triangles: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    return _proper_segment_triangle_hits(
        start, end, triangles, tolerance
    )[0]


def _segment_triangle_distance(
    start: np.ndarray,
    end: np.ndarray,
    triangles: np.ndarray,
    tolerance: float,
) -> float:
    if len(
        _proper_segment_triangle_parameters(
            start, end, triangles, tolerance
        )
    ):
        return 0.0
    distance = min(
        _point_triangle_distance(start, triangles),
        _point_triangle_distance(end, triangles),
    )
    for first, second in ((0, 1), (1, 2), (2, 0)):
        distance = min(
            distance,
            float(
                np.min(
                    _segment_segment_distance_to_many(
                        start,
                        end,
                        triangles[:, first],
                        triangles[:, second],
                    ),
                    initial=np.inf,
                )
            ),
        )
    return distance


def _point_class(
    point: np.ndarray,
    triangles: np.ndarray,
    tolerance: float,
) -> str:
    if _point_triangle_distance(point, triangles) <= tolerance:
        return "boundary"
    relative = triangles - point
    norm = np.linalg.norm(relative, axis=2)
    numerator = np.einsum(
        "ij,ij->i",
        relative[:, 0],
        np.cross(relative[:, 1], relative[:, 2]),
    )
    denominator = (
        np.prod(norm, axis=1)
        + np.einsum("ij,ij->i", relative[:, 0], relative[:, 1])
        * norm[:, 2]
        + np.einsum("ij,ij->i", relative[:, 1], relative[:, 2])
        * norm[:, 0]
        + np.einsum("ij,ij->i", relative[:, 2], relative[:, 0])
        * norm[:, 1]
    )
    solid_angle = float(np.sum(2.0 * np.arctan2(numerator, denominator)))
    return "inside" if abs(solid_angle) > 2.0 * np.pi else "outside"


def _active_unique_segments(
    rings: np.ndarray,
    gamma: np.ndarray,
    topology_tolerance: float,
    circulation_tolerance: float,
) -> tuple[int, int, list[tuple[np.ndarray, np.ndarray, float]]]:
    accumulators: dict[
        tuple[tuple[int, ...], tuple[int, ...]],
        list[Any],
    ] = {}
    for ring, strength in zip(rings, gamma, strict=True):
        for first, second in ((0, 1), (1, 2), (2, 3), (3, 0)):
            start = ring[first]
            end = ring[second]
            start_key = tuple(
                np.rint(start / topology_tolerance).astype(np.int64)
            )
            end_key = tuple(
                np.rint(end / topology_tolerance).astype(np.int64)
            )
            if start_key <= end_key:
                key = (start_key, end_key)
                signed = float(strength)
                canonical_start = start
                canonical_end = end
            else:
                key = (end_key, start_key)
                signed = -float(strength)
                canonical_start = end
                canonical_end = start
            if key not in accumulators:
                accumulators[key] = [
                    canonical_start.copy(),
                    canonical_end.copy(),
                    signed,
                ]
            else:
                accumulators[key][2] += signed
    active = [
        (value[0], value[1], float(value[2]))
        for value in accumulators.values()
        if abs(float(value[2])) > circulation_tolerance
    ]
    return len(accumulators), len(accumulators) - len(active), active


def _audit_filament_channel(
    channel: str,
    rings: np.ndarray,
    gamma: np.ndarray,
    mesh: ClosedTriangularMesh,
    face_roles: np.ndarray | None = None,
) -> FilamentChannelTopology:
    triangles = mesh.vertices[mesh.faces]
    scale = max(
        float(np.ptp(mesh.vertices, axis=0).max(initial=0.0)),
        1.0,
    )
    topology_tolerance = 4096.0 * np.finfo(float).eps * scale
    circulation_scale = max(
        float(np.max(np.abs(gamma), initial=0.0)), 1.0
    )
    circulation_tolerance = (
        4096.0 * np.finfo(float).eps * circulation_scale
    )
    unique_count, cancelled_count, segments = _active_unique_segments(
        rings,
        gamma,
        topology_tolerance,
        circulation_tolerance,
    )
    inside_inside = 0
    outside_outside = 0
    inside_outside = 0
    boundary_endpoint = 0
    contact = 0
    piercing = 0
    intersection_count = 0
    minimum_distance = np.inf
    total_weight = 0.0
    piercing_weight = 0.0
    examples: list[ActiveFilament] = []
    face_role_counts: dict[str, int] = {}
    if face_roles is not None:
        roles = np.asarray(face_roles)
        if roles.shape != (len(mesh.faces),):
            raise N1ThickShellAdapterError(
                "face_roles must contain one role per shell face"
            )
    else:
        roles = np.full(len(mesh.faces), "unlabelled", dtype=object)
    for start, end, circulation in segments:
        start_class = _point_class(
            start, triangles, topology_tolerance
        )
        end_class = _point_class(end, triangles, topology_tolerance)
        classes = (start_class, end_class)
        if classes == ("inside", "inside"):
            inside_inside += 1
        elif classes == ("outside", "outside"):
            outside_outside += 1
        elif "boundary" in classes:
            boundary_endpoint += 1
        else:
            inside_outside += 1
        parameters, face_groups = _proper_segment_triangle_hits(
            start, end, triangles, topology_tolerance
        )
        distance = _segment_triangle_distance(
            start, end, triangles, topology_tolerance
        )
        minimum_distance = min(minimum_distance, distance)
        if distance <= topology_tolerance:
            contact += 1
        count = int(len(parameters))
        weight = abs(circulation)
        total_weight += weight
        if count:
            piercing += 1
            intersection_count += count
            piercing_weight += weight
            intersection_roles: list[str] = []
            for face_group in face_groups:
                group_roles = sorted({
                    str(roles[index]) for index in face_group
                })
                for role in group_roles:
                    face_role_counts[role] = (
                        face_role_counts.get(role, 0) + 1
                    )
                    intersection_roles.append(role)
            if len(examples) < 12:
                examples.append(
                    ActiveFilament(
                        start=start.copy(),
                        end=end.copy(),
                        circulation=circulation,
                        endpoint_classes=classes,
                        proper_intersection_count=count,
                        intersection_face_roles=tuple(
                            intersection_roles
                        ),
                        minimum_shell_distance=distance,
                    )
                )
    return FilamentChannelTopology(
        channel=channel,
        raw_segment_count=4 * len(rings),
        unique_segment_count=unique_count,
        active_unique_segment_count=len(segments),
        cancelled_unique_segment_count=cancelled_count,
        inside_inside_segment_count=inside_inside,
        outside_outside_segment_count=outside_outside,
        inside_outside_segment_count=inside_outside,
        boundary_endpoint_segment_count=boundary_endpoint,
        shell_contact_segment_count=contact,
        proper_shell_piercing_segment_count=piercing,
        proper_shell_intersection_count=intersection_count,
        proper_intersection_face_role_counts=tuple(
            sorted(face_role_counts.items())
        ),
        minimum_active_segment_to_shell_distance=(
            float(minimum_distance) if segments else None
        ),
        circulation_weighted_piercing_fraction=(
            piercing_weight / max(total_weight, np.finfo(float).tiny)
        ),
        topology_tolerance=topology_tolerance,
        circulation_tolerance=circulation_tolerance,
        piercing_examples=tuple(examples),
    )


def audit_n1_filament_shell_topology(
    snapshot: N1StepSnapshot,
    shell: N1ActualShellKinematics,
) -> N1FilamentShellTopologyAudit:
    """Classify frozen singular support without moving or coring it."""
    if not isinstance(snapshot, N1StepSnapshot):
        raise N1ThickShellAdapterError(
            "snapshot must be an N1StepSnapshot"
        )
    if not isinstance(shell, N1ActualShellKinematics):
        raise N1ThickShellAdapterError(
            "shell must be an N1ActualShellKinematics"
        )
    bound_image = mirrored_ring_field(snapshot.bound_rings)
    wake_image = mirrored_ring_field(snapshot.wake_rings)
    mesh = shell.closed_shell.mesh
    return N1FilamentShellTopologyAudit(
        bound_direct=_audit_filament_channel(
            "bound_direct",
            snapshot.bound_rings,
            snapshot.bound_gamma,
            mesh,
            shell.closed_shell.face_roles,
        ),
        bound_image=_audit_filament_channel(
            "bound_image",
            bound_image,
            snapshot.bound_gamma,
            mesh,
            shell.closed_shell.face_roles,
        ),
        wake_direct=_audit_filament_channel(
            "wake_direct",
            snapshot.wake_rings,
            snapshot.wake_gamma,
            mesh,
            shell.closed_shell.face_roles,
        ),
        wake_image_candidate=_audit_filament_channel(
            "wake_image_candidate",
            wake_image,
            snapshot.wake_gamma,
            mesh,
            shell.closed_shell.face_roles,
        ),
    )


def trailing_edge_topology_residual(
    snapshot: N1StepSnapshot,
    shell: N1ActualShellKinematics,
    *,
    dt: float,
) -> TrailingEdgeTopologyResidual:
    """Compare actual upper/base/lower topology with the N1 rear ring line."""
    if not isinstance(snapshot, N1StepSnapshot):
        raise N1ThickShellAdapterError(
            "snapshot must be an N1StepSnapshot"
        )
    if not isinstance(shell, N1ActualShellKinematics):
        raise N1ThickShellAdapterError(
            "shell must be an N1ActualShellKinematics"
        )
    spacing = float(dt)
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise N1ThickShellAdapterError("dt must be positive and finite")
    material = shell.dual_surface_shell
    if material.mean_surface.shape != snapshot.mean_surface.shape:
        raise N1ThickShellAdapterError(
            "topology residual requires the shell on the N1 material grid"
        )
    if snapshot.nc + 1 != material.mean_surface.shape[0]:
        raise N1ThickShellAdapterError("N1 chord topology mismatch")
    if snapshot.ns + 1 != material.mean_surface.shape[1]:
        raise N1ThickShellAdapterError("N1 span topology mismatch")

    mean = material.mean_surface
    upper = material.upper_surface
    lower = material.lower_surface
    chord_endpoint = np.linalg.norm(mean[-1] - mean[0], axis=1)
    chord = 0.5 * (chord_endpoint[:-1] + chord_endpoint[1:])
    base_nodes = np.linalg.norm(upper[-1] - lower[-1], axis=1)
    base = 0.5 * (base_nodes[:-1] + base_nodes[1:])

    last_row = snapshot.bound_rings[
        (snapshot.nc - 1) * snapshot.ns:
        snapshot.nc * snapshot.ns
    ]
    rear_left = last_row[:, 3]
    rear_right = last_row[:, 2]
    trailing = mean[-1]
    offset_left = rear_left - trailing[:-1]
    offset_right = rear_right - trailing[1:]
    offset = 0.5 * (
        np.linalg.norm(offset_left, axis=1)
        + np.linalg.norm(offset_right, axis=1)
    )
    expected_left = (
        trailing[:-1]
        + 0.25 * (trailing[:-1] - mean[-2, :-1])
    )
    expected_right = (
        trailing[1:]
        + 0.25 * (trailing[1:] - mean[-2, 1:])
    )
    identity_error = float(
        max(
            np.max(np.abs(rear_left - expected_left), initial=0.0),
            np.max(np.abs(rear_right - expected_right), initial=0.0),
        )
    )
    freestream_step = (
        float(np.linalg.norm(snapshot.freestream_velocity)) * spacing
    )
    return TrailingEdgeTopologyResidual(
        base_thickness=base,
        base_thickness_over_chord=base / chord,
        n1_rear_line_offset=offset,
        n1_rear_line_offset_over_chord=offset / chord,
        rear_line_kernel_identity_error=identity_error,
        base_to_rear_offset_ratio=base / np.maximum(
            offset, np.finfo(float).tiny
        ),
        first_wake_step_over_chord=np.full_like(chord, freestream_step)
        / chord,
        actual_topology="upper_corner+solid_base+lower_corner",
        frozen_n1_topology=(
            "single_mean_surface_rear_ring_line_plus_single_wake_sheet"
        ),
    )


def _face_index(
    lookup: dict[tuple[int, int, int], int],
    vertices: tuple[int, int, int],
) -> int:
    key = tuple(sorted(vertices))
    try:
        return lookup[key]
    except KeyError as error:
        raise N1ThickShellAdapterError(
            f"cannot identify closed-shell face {vertices}"
        ) from error


def trailing_edge_direction_residual(
    shell: N1ActualShellKinematics,
    source: NeumannSourceSolution,
) -> TrailingEdgeDirectionResidual:
    """Report B-1/B-2 direction residuals without selecting one as truth."""
    if not isinstance(shell, N1ActualShellKinematics):
        raise N1ThickShellAdapterError(
            "shell must be an N1ActualShellKinematics"
        )
    if not isinstance(source, NeumannSourceSolution):
        raise N1ThickShellAdapterError(
            "source must be a NeumannSourceSolution"
        )
    closed = shell.closed_shell
    if (
        source.mesh.vertices.shape != closed.mesh.vertices.shape
        or source.mesh.faces.shape != closed.mesh.faces.shape
        or not np.array_equal(source.mesh.faces, closed.mesh.faces)
        or not np.array_equal(source.mesh.vertices, closed.mesh.vertices)
    ):
        raise N1ThickShellAdapterError(
            "source solution does not belong to this closed shell"
        )
    upper_indices = closed.upper_vertex_indices
    lower_indices = closed.lower_vertex_indices
    chord_cell = upper_indices.shape[0] - 2
    span_cells = upper_indices.shape[1] - 1
    lookup = {
        tuple(sorted(map(int, face))): index
        for index, face in enumerate(closed.mesh.faces)
    }
    upper_faces = np.empty((span_cells, 2), dtype=int)
    lower_faces = np.empty((span_cells, 2), dtype=int)
    for span_cell in range(span_cells):
        u00 = int(upper_indices[chord_cell, span_cell])
        u10 = int(upper_indices[chord_cell + 1, span_cell])
        u11 = int(
            upper_indices[chord_cell + 1, span_cell + 1]
        )
        u01 = int(upper_indices[chord_cell, span_cell + 1])
        upper_faces[span_cell] = (
            _face_index(lookup, (u00, u10, u11)),
            _face_index(lookup, (u00, u11, u01)),
        )
        l00 = int(lower_indices[chord_cell, span_cell])
        l10 = int(lower_indices[chord_cell + 1, span_cell])
        l11 = int(
            lower_indices[chord_cell + 1, span_cell + 1]
        )
        l01 = int(lower_indices[chord_cell, span_cell + 1])
        lower_faces[span_cell] = (
            _face_index(lookup, (l00, l11, l10)),
            _face_index(lookup, (l00, l01, l11)),
        )

    def area_average(indices, values):
        areas = source.mesh.areas[indices]
        return np.einsum(
            "ij,ijk->ik", areas, values[indices]
        ) / np.sum(areas, axis=1)[:, None]

    relative_velocity = source.total_velocity - source.wall_velocity
    upper_velocity = area_average(upper_faces, relative_velocity)
    lower_velocity = area_average(lower_faces, relative_velocity)
    material = shell.dual_surface_shell
    upper = material.upper_surface
    lower = material.lower_surface
    upper_tangent = _unit(
        "upper trailing tangent",
        0.5 * (upper[-1, :-1] + upper[-1, 1:])
        - 0.5 * (upper[-2, :-1] + upper[-2, 1:]),
    )
    lower_tangent = _unit(
        "lower trailing tangent",
        0.5 * (lower[-1, :-1] + lower[-1, 1:])
        - 0.5 * (lower[-2, :-1] + lower[-2, 1:]),
    )
    trailing_midline = 0.5 * (upper[-1] + lower[-1])
    span_tangent = _unit(
        "trailing span tangent",
        trailing_midline[1:] - trailing_midline[:-1],
    )

    def projected_unit(name, tangent):
        projected = tangent - np.einsum(
            "ij,ij->i", tangent, span_tangent
        )[:, None] * span_tangent
        return _unit(name, projected)

    upper_chord = projected_unit("upper chord tangent", upper_tangent)
    lower_chord = projected_unit("lower chord tangent", lower_tangent)
    bisector = projected_unit(
        "trailing-edge bisector", upper_chord + lower_chord
    )
    upper_normal = _unit(
        "upper in-plane normal", np.cross(span_tangent, upper_chord)
    )
    lower_normal = _unit(
        "lower in-plane normal", np.cross(span_tangent, lower_chord)
    )
    bisector_normal = _unit(
        "bisector in-plane normal", np.cross(span_tangent, bisector)
    )
    mean_velocity = 0.5 * (upper_velocity + lower_velocity)
    return TrailingEdgeDirectionResidual(
        upper_face_indices=upper_faces,
        lower_face_indices=lower_faces,
        upper_tangent_normal_velocity=np.einsum(
            "ij,ij->i", upper_velocity, upper_normal
        ),
        lower_tangent_normal_velocity=np.einsum(
            "ij,ij->i", lower_velocity, lower_normal
        ),
        bisector_normal_velocity=np.einsum(
            "ij,ij->i", mean_velocity, bisector_normal
        ),
        upper_relative_speed=np.linalg.norm(upper_velocity, axis=1),
        lower_relative_speed=np.linalg.norm(lower_velocity, axis=1),
        source_flux=source.source_flux,
        relative_source_flux=source.relative_source_flux,
        pressure_residual_available=False,
        pressure_residual_blocker=(
            "qualified total material-potential history is absent"
        ),
    )
