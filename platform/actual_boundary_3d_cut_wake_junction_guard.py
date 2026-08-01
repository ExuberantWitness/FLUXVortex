"""Run the preregistered S3a 3-D body-cut/material-wake junction oracle."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from claim_runtime.classified_p2_cut_topology import (  # noqa: E402
    ClassifiedP2CutTopology,
    classified_p2_cut_topology,
)
from claim_runtime.distributed_doublet import (  # noqa: E402
    newborn_material_wake_band,
)
from claim_runtime.n1_dde_interface import (  # noqa: E402
    dde_mu_to_n1_gamma,
    n1_gamma_to_dde_mu,
)
from claim_runtime.thick_body_neumann_shadow import (  # noqa: E402
    ClosedTriangularMesh,
    closed_triangular_mesh,
)


CASES = (
    HERE / "docs" / "diag"
    / "actual_boundary_3d_cut_wake_junction_cases.yaml"
)
RESULTS = (
    HERE / "docs" / "diag"
    / "actual_boundary_3d_cut_wake_junction_results.json"
)


def _oriented_triangle(
    vertices: np.ndarray,
    face: tuple[int, int, int],
    outward: np.ndarray,
) -> tuple[int, int, int]:
    first, second, third = face
    normal = np.cross(
        vertices[second] - vertices[first],
        vertices[third] - vertices[first],
    )
    if float(np.dot(normal, outward)) < 0.0:
        return first, third, second
    return face


def build_canonical_diamond_wing() -> tuple[
    ClosedTriangularMesh,
    np.ndarray,
    np.ndarray,
    tuple[tuple[int, int], ...],
    tuple[int, int],
]:
    """Build the frozen full-wing finite-angle S3a body canonical."""
    section = np.array(
        (
            (0.0, 0.0),
            (0.5, 0.1),
            (1.0, 0.0),
            (0.5, -0.1),
        ),
        dtype=float,
    )
    span = np.linspace(-1.0, 1.0, 5)
    vertices = np.array(
        [
            (x, y, z)
            for y in span
            for x, z in section
        ],
        dtype=float,
    )

    def vertex(span_index: int, section_index: int) -> int:
        return 4 * span_index + section_index

    faces: list[tuple[int, int, int]] = []
    upper_faces: list[int] = []
    lower_faces: list[int] = []
    for span_index in range(len(span) - 1):
        for section_index in range(4):
            following = (section_index + 1) % 4
            first = vertex(span_index, section_index)
            second = vertex(span_index, following)
            third = vertex(span_index + 1, following)
            fourth = vertex(span_index + 1, section_index)
            centroid = np.mean(
                vertices[[first, second, third, fourth]],
                axis=0,
            )
            outward = np.array(
                (centroid[0] - 0.5, 0.0, centroid[2]),
                dtype=float,
            )
            negative_span = (
                0.5 * (span[span_index] + span[span_index + 1])
                < 0.0
            )
            lower_side = section_index >= 2
            if negative_span != lower_side:
                raw_pair = (
                    (first, second, third),
                    (first, third, fourth),
                )
            else:
                raw_pair = (
                    (first, second, fourth),
                    (second, third, fourth),
                )
            pair = tuple(
                _oriented_triangle(vertices, triangle, outward)
                for triangle in raw_pair
            )
            owner = upper_faces if section_index < 2 else lower_faces
            for triangle in pair:
                owner.append(len(faces))
                faces.append(triangle)

    for span_index, outward_y in ((0, -1.0), (len(span) - 1, 1.0)):
        le = vertex(span_index, 0)
        upper = vertex(span_index, 1)
        trailing = vertex(span_index, 2)
        lower = vertex(span_index, 3)
        for triangle in ((le, upper, trailing), (le, trailing, lower)):
            faces.append(
                _oriented_triangle(
                    vertices,
                    triangle,
                    np.array((0.0, outward_y, 0.0)),
                )
            )

    mesh = closed_triangular_mesh(
        vertices,
        np.asarray(faces, dtype=np.int64),
    )
    trailing_vertices = tuple(
        vertex(span_index, 2)
        for span_index in range(len(span))
    )
    cut_edges = tuple(
        (first, second)
        for first, second in zip(
            trailing_vertices,
            trailing_vertices[1:],
        )
    )
    return (
        mesh,
        np.asarray(upper_faces, dtype=np.int64),
        np.asarray(lower_faces, dtype=np.int64),
        cut_edges,
        (trailing_vertices[0], trailing_vertices[-1]),
    )


def _quadratic_derivative_at_center(
    coordinate: np.ndarray,
    values: np.ndarray,
) -> float:
    center = int(np.argmin(np.abs(coordinate)))
    left = np.polyfit(
        coordinate[center - 2 : center + 1],
        values[center - 2 : center + 1],
        deg=2,
    )
    right = np.polyfit(
        coordinate[center : center + 3],
        values[center : center + 3],
        deg=2,
    )
    return float(max(abs(left[1]), abs(right[1])))


def _topology_record(
    topology: ClassifiedP2CutTopology,
) -> dict:
    return {
        "base_dof_count": topology.base_topology.dof_count,
        "cut_dof_count": topology.dof_count,
        "duplicated_dof_count": topology.duplicated_dof_count,
        "duplicated_vertex_count": (
            topology.duplicated_vertex_count
        ),
        "duplicated_edge_midpoint_count": (
            topology.duplicated_edge_midpoint_count
        ),
        "cut_edges": topology.cut_edges.tolist(),
        "ordered_cut_vertex_indices": (
            topology.ordered_cut_vertex_indices.tolist()
        ),
        "cut_node_coordinates": (
            topology.cut_node_coordinates.tolist()
        ),
        "upper_cut_dofs": topology.upper_cut_dofs.tolist(),
        "lower_cut_dofs": topology.lower_cut_dofs.tolist(),
        "noncut_trace_dof_mismatch_count": (
            topology.noncut_trace_dof_mismatch_count
        ),
        "maximum_cut_coordinate_pair_gap": (
            topology.maximum_cut_coordinate_pair_gap
        ),
    }


def run() -> dict:
    contract = yaml.safe_load(CASES.read_text(encoding="utf-8"))
    thresholds = contract["thresholds"]
    (
        mesh,
        upper_faces,
        lower_faces,
        cut_edges,
        zero_endpoints,
    ) = build_canonical_diamond_wing()
    physical_vertices_before = mesh.vertices.copy()
    topology = classified_p2_cut_topology(
        mesh,
        upper_face_indices=upper_faces,
        lower_face_indices=lower_faces,
        cut_edges=cut_edges,
        zero_jump_end_vertices=zero_endpoints,
    )

    span_coordinate = topology.cut_node_coordinates[:, 1]
    prescribed_mu = 1.0 - span_coordinate**2
    trace = np.zeros(topology.dof_count, dtype=float)
    trace[topology.upper_cut_dofs] = 0.5 * prescribed_mu
    trace[topology.lower_cut_dofs] = -0.5 * prescribed_mu
    body_jump = topology.cut_jump(trace)
    body_jump_error = float(
        np.max(np.abs(body_jump - prescribed_mu), initial=0.0)
    )

    gauge_shift = float(contract["canonical"]["gauge_shift"])
    shifted_jump = topology.cut_jump(trace + gauge_shift)
    gauge_jump_change = float(
        np.max(np.abs(shifted_jump - body_jump), initial=0.0)
    )

    span_vertices = np.linspace(-1.0, 1.0, 5)
    current_edge = np.column_stack(
        (
            np.ones_like(span_vertices),
            span_vertices,
            np.zeros_like(span_vertices),
        )
    )
    previous_edge = current_edge.copy()
    previous_edge[:, 0] = 1.1
    rows = np.repeat(prescribed_mu[None, :], 3, axis=0)
    wake = newborn_material_wake_band(
        sheet_id="S3a-TEV-row-0",
        vortex_family="TEV",
        previous_edge=previous_edge,
        current_edge=current_edge,
        time_nodes=np.array((0.0, 0.5, 1.0)),
        potential_jump_rows=rows,
    )
    wake_attachment_error = float(
        np.max(
            np.abs(wake.potential_jump_rows[-1] - body_jump),
            initial=0.0,
        )
    )
    gauge_wake_attachment_error = float(
        np.max(
            np.abs(wake.potential_jump_rows[-1] - shifted_jump),
            initial=0.0,
        )
    )

    interface_normal = np.repeat(
        np.array(((0.0, 0.0, 1.0),)),
        len(prescribed_mu),
        axis=0,
    )
    gamma_n1 = -prescribed_mu
    mapped_mu = n1_gamma_to_dde_mu(
        gamma_n1,
        n1_normal=interface_normal,
        dde_normal=interface_normal,
    )
    roundtrip_gamma = dde_mu_to_n1_gamma(
        mapped_mu,
        n1_normal=interface_normal,
        dde_normal=interface_normal,
    )
    interface_roundtrip_error = float(max(
        np.max(np.abs(mapped_mu - prescribed_mu), initial=0.0),
        np.max(np.abs(roundtrip_gamma - gamma_n1), initial=0.0),
    ))

    geometry_change = float(
        np.max(
            np.abs(mesh.vertices - physical_vertices_before),
            initial=0.0,
        )
    )
    tip_jump = float(max(abs(body_jump[0]), abs(body_jump[-1])))
    symmetry_derivative = _quadratic_derivative_at_center(
        span_coordinate,
        body_jump,
    )
    wake_continuity = wake.surface.continuity_report(
        tolerance=float(
            thresholds["wake_internal_trace_jump_abs_max"]
        )
    )
    wake_internal_jump = float(max(
        wake_continuity.max_trace_node_jump,
        wake_continuity.max_trace_jump,
    ))
    expected_duplicate_count = (
        (len(span_vertices) - 2) + (len(span_vertices) - 1)
    )

    checks = {
        "body_geometry_unchanged": geometry_change
        <= float(thresholds["body_geometry_change_abs_max"]),
        "body_remains_watertight": (
            mesh.boundary_edge_count
            <= int(thresholds["body_boundary_edge_count_max"])
            and mesh.nonmanifold_edge_count
            <= int(thresholds["body_nonmanifold_edge_count_max"])
            and mesh.orientation_mismatch_count == 0
        ),
        "only_internal_cut_dofs_duplicated": (
            topology.duplicated_vertex_count
            == len(span_vertices) - 2
            and topology.duplicated_edge_midpoint_count
            == len(span_vertices) - 1
            and topology.duplicated_dof_count
            == expected_duplicate_count
            and topology.dof_count
            == topology.base_topology.dof_count
            + expected_duplicate_count
        ),
        "noncut_trace_remains_shared": (
            topology.noncut_trace_dof_mismatch_count
            <= int(thresholds["noncut_trace_dof_mismatch_max"])
        ),
        "cut_copies_are_coincident": (
            topology.maximum_cut_coordinate_pair_gap
            <= float(
                thresholds["cut_coordinate_pair_gap_abs_max"]
            )
        ),
        "body_jump_matches_manufactured_trace": (
            body_jump_error
            <= float(thresholds["body_jump_abs_error_max"])
        ),
        "wake_current_edge_matches_body_jump": (
            wake_attachment_error
            <= float(
                thresholds["wake_attachment_abs_error_max"]
            )
        ),
        "gauge_invariance": (
            max(gauge_jump_change, gauge_wake_attachment_error)
            <= float(
                thresholds["gauge_jump_change_abs_max"]
            )
        ),
        "n1_dde_sign_roundtrip": (
            interface_roundtrip_error
            <= float(
                thresholds["interface_roundtrip_abs_error_max"]
            )
        ),
        "tip_jump_is_zero": (
            tip_jump
            <= float(thresholds["tip_jump_abs_max"])
        ),
        "full_wing_symmetry_derivative_is_zero": (
            symmetry_derivative
            <= float(
                thresholds["symmetry_derivative_abs_max"]
            )
        ),
        "wake_is_internally_p2_continuous": (
            wake_continuity.compatible
            and wake_internal_jump
            <= float(
                thresholds["wake_internal_trace_jump_abs_max"]
            )
        ),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    result = {
        "artifact": "actual_boundary_3d_cut_wake_junction_oracle",
        "claim_node": contract["claim_node"],
        "stage": contract["stage"],
        "canonical": contract["canonical"],
        "mesh": {
            "vertices": len(mesh.vertices),
            "faces": len(mesh.faces),
            "signed_volume": mesh.signed_volume,
            "boundary_edge_count": mesh.boundary_edge_count,
            "nonmanifold_edge_count": mesh.nonmanifold_edge_count,
            "orientation_mismatch_count": (
                mesh.orientation_mismatch_count
            ),
        },
        "topology": _topology_record(topology),
        "aggregate_metrics": {
            "body_geometry_change_abs_max": geometry_change,
            "body_jump_abs_error_max": body_jump_error,
            "wake_attachment_abs_error_max": wake_attachment_error,
            "gauge_jump_change_abs_max": gauge_jump_change,
            "gauge_wake_attachment_abs_error_max": (
                gauge_wake_attachment_error
            ),
            "interface_roundtrip_abs_error_max": (
                interface_roundtrip_error
            ),
            "tip_jump_abs_max": tip_jump,
            "symmetry_derivative_abs_max": symmetry_derivative,
            "wake_internal_trace_jump_abs_max": wake_internal_jump,
        },
        "thresholds": thresholds,
        "checks": checks,
        "stage_decision": (
            "GO" if all(checks.values()) else "NO-GO"
        ),
        "production_activation_allowed": False,
        "interpretation": (
            "Passing validates only that a watertight 3-D body can carry "
            "a classified quadratic potential cut whose jump is the same "
            "continuous material quantity on the newborn wake edge.  It "
            "does not select circulation, solve body/wake equations, close "
            "a finite blunt base, compute pressure or activate force."
        ),
    }
    RESULTS.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
