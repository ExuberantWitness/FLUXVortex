"""Preregistered guard for the frozen-N1 actual-thickness-shell adapter."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import warp as wp

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import diff_uvlm_unsteady_gpu as ug  # noqa: E402
from _v2_robo import (  # noqa: E402
    _f3,
    _ring_vel_f,
    gpu_run_twist,
    twisted_corners,
)
import _v2_robogeom as rg  # noqa: E402
from claim_runtime.hirato_shadow import mirrored_ring_field  # noqa: E402
from claim_runtime.n1_thick_shell_adapter import (  # noqa: E402
    N1StepSnapshot,
    build_actual_shell_kinematics_from_mean_surfaces,
    build_n1_actual_shell_kinematics,
    evaluate_n1_incident_velocity,
    parse_n1_snapshot_triplet,
    production_ring_field_velocity,
    replay_n1_collocation_boundary,
)
from claim_runtime.thick_body_trailing_edge import (  # noqa: E402
    audit_n1_filament_shell_topology,
    closed_surface_flux_ledger,
    trailing_edge_direction_residual,
    trailing_edge_topology_residual,
)
from claim_runtime.thick_body_neumann_shadow import (  # noqa: E402
    solve_conditioned_neumann_source,
)


V3 = wp.vec3d
V3F = wp.vec3f
DTYPE = wp.float64


@wp.kernel
def _manufactured_ring_field_fp64(
    points: wp.array(dtype=V3),
    rings: wp.array(dtype=V3, ndim=2),
    gamma: wp.array(dtype=DTYPE),
    ring_count: int,
    velocity: wp.array(dtype=V3),
):
    point_index = wp.tid()
    result = V3(0.0, 0.0, 0.0)
    for ring_index in range(ring_count):
        result = result + gamma[ring_index] * ug.ring_vel(
            points[point_index],
            rings[ring_index, 0],
            rings[ring_index, 1],
            rings[ring_index, 2],
            rings[ring_index, 3],
        )
    velocity[point_index] = result


@wp.kernel
def _manufactured_ring_field_fp32(
    points: wp.array(dtype=V3),
    rings: wp.array(dtype=V3, ndim=2),
    gamma: wp.array(dtype=DTYPE),
    ring_count: int,
    velocity: wp.array(dtype=V3),
):
    point_index = wp.tid()
    point = _f3(points[point_index])
    result = V3F(0.0, 0.0, 0.0)
    for ring_index in range(ring_count):
        result = result + wp.float32(gamma[ring_index]) * _ring_vel_f(
            point,
            _f3(rings[ring_index, 0]),
            _f3(rings[ring_index, 1]),
            _f3(rings[ring_index, 2]),
            _f3(rings[ring_index, 3]),
        )
    velocity[point_index] = V3(
        wp.float64(result[0]),
        wp.float64(result[1]),
        wp.float64(result[2]),
    )


def _relative_vector_error(actual, expected) -> float:
    numerator = float(
        np.max(np.linalg.norm(actual - expected, axis=1), initial=0.0)
    )
    denominator = max(
        float(np.max(np.linalg.norm(expected, axis=1), initial=0.0)),
        np.finfo(float).tiny,
    )
    return numerator / denominator


def _manufactured_oracle() -> dict:
    rings = np.array(
        [
            [
                [0.0, 0.17, 0.02],
                [0.0, 0.83, 0.05],
                [0.61, 0.82, 0.11],
                [0.58, 0.18, 0.08],
            ],
            [
                [0.3, 0.25, -0.11],
                [0.4, 0.72, -0.07],
                [0.9, 0.75, 0.03],
                [0.8, 0.21, -0.02],
            ],
        ],
        dtype=float,
    )
    rings = np.concatenate((rings, mirrored_ring_field(rings)), axis=0)
    gamma = np.array([0.37, -0.19, 0.37, -0.19])
    points = np.array(
        [
            [-0.3, 0.31, 0.63],
            [0.22, 0.47, 0.91],
            [1.37, -0.28, 0.52],
            [0.71, 1.42, -0.44],
        ]
    )
    device = "cuda:0"
    point_wp = wp.array(points, dtype=V3, device=device)
    ring_wp = wp.array(rings, dtype=V3, device=device)
    gamma_wp = wp.array(gamma, dtype=DTYPE, device=device)
    oracle64 = wp.zeros(len(points), dtype=V3, device=device)
    oracle32 = wp.zeros(len(points), dtype=V3, device=device)
    wp.launch(
        _manufactured_ring_field_fp64,
        dim=len(points),
        inputs=[point_wp, ring_wp, gamma_wp, len(rings)],
        outputs=[oracle64],
        device=device,
    )
    wp.launch(
        _manufactured_ring_field_fp32,
        dim=len(points),
        inputs=[point_wp, ring_wp, gamma_wp, len(rings)],
        outputs=[oracle32],
        device=device,
    )
    actual64 = production_ring_field_velocity(
        points,
        rings,
        gamma,
        arithmetic="fp64",
        denominator_floor=1.0e-10,
    )
    actual32 = production_ring_field_velocity(
        points,
        rings,
        gamma,
        arithmetic="fp32",
        denominator_floor=1.0e-10,
    )
    error64 = _relative_vector_error(actual64, oracle64.numpy())
    error32 = _relative_vector_error(actual32, oracle32.numpy())
    return {
        "ring_count_including_images": len(rings),
        "point_count": len(points),
        "fp64_velocity_max_relative_error": error64,
        "fp32_velocity_max_relative_error": error32,
        "fp64_gate": error64 <= 2.0e-12,
        "fp32_gate": error32 <= 2.0e-5,
    }


def _hash_frame_arrays(frames) -> str:
    digest = hashlib.sha256()
    for frame in frames:
        for key in sorted(frame):
            value = frame[key]
            digest.update(key.encode("utf-8"))
            if isinstance(value, np.ndarray):
                digest.update(value.dtype.str.encode("ascii"))
                digest.update(str(value.shape).encode("ascii"))
                digest.update(value.tobytes(order="C"))
            else:
                digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()


def _numeric_results_equal(first, second) -> tuple[bool, list[str]]:
    failures = []
    common = sorted(set(first) & set(second))
    for key in common:
        left = first[key]
        right = second[key]
        if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
            if not np.array_equal(left, right, equal_nan=True):
                failures.append(key)
        elif isinstance(left, (int, float, np.number)) and isinstance(
            right, (int, float, np.number)
        ):
            if not np.array_equal(
                np.asarray(left), np.asarray(right), equal_nan=True
            ):
                failures.append(key)
    return not failures, failures


def _flux_ledger_dict(ledger) -> dict[str, float]:
    return {
        name: float(getattr(ledger, name))
        for name in (
            "freestream",
            "bound_direct",
            "bound_image",
            "wake_direct",
            "wake_image_candidate",
            "production_total",
            "physical_symmetry_candidate_total",
            "wall_volume_flux",
            "production_reconstruction_error",
            "physical_symmetry_reconstruction_error",
        )
    }


def _filament_channel_dict(channel) -> dict:
    return {
        name: getattr(channel, name)
        for name in (
            "channel",
            "raw_segment_count",
            "unique_segment_count",
            "active_unique_segment_count",
            "cancelled_unique_segment_count",
            "inside_inside_segment_count",
            "outside_outside_segment_count",
            "inside_outside_segment_count",
            "boundary_endpoint_segment_count",
            "shell_contact_segment_count",
            "proper_shell_piercing_segment_count",
            "proper_shell_intersection_count",
            "proper_intersection_face_role_counts",
            "minimum_active_segment_to_shell_distance",
            "circulation_weighted_piercing_fraction",
            "topology_tolerance",
            "circulation_tolerance",
        )
    } | {
        "piercing_examples": [
            {
                "start": example.start.tolist(),
                "end": example.end.tolist(),
                "circulation": float(example.circulation),
                "endpoint_classes": list(example.endpoint_classes),
                "proper_intersection_count":
                    example.proper_intersection_count,
                "intersection_face_roles":
                    list(example.intersection_face_roles),
                "minimum_shell_distance":
                    example.minimum_shell_distance,
            }
            for example in channel.piercing_examples
        ]
    }


def _representative_case() -> dict:
    arguments = dict(
        U=8.0,
        aoa_deg=15.0,
        freq=2.6,
        flap_amp_deg=22.5,
        twist_amp_deg=11.25,
        twist_phase_deg=90.0,
        nc=4,
        ns=8,
        n_cycle=2,
        steps_per_cycle=60,
        wake_rows=60,
        real_geom=True,
        sym=True,
        les_suction=True,
        visc=True,
        d_para=0.5,
        a0_crit=0.23,
        closure="v41",
        lev_shed_mode="none",
        tev_core=0.0,
    )
    baseline = gpu_run_twist(**arguments)
    frames = []
    observed = gpu_run_twist(
        **arguments,
        frames_out=frames,
        frame_skip=1,
    )
    numerical_unchanged, numerical_failures = _numeric_results_equal(
        baseline, observed
    )
    before_hash = _hash_frame_arrays(frames)
    triplet = parse_n1_snapshot_triplet(
        frames[-3:],
        expected_dt=1.0 / arguments["freq"] / arguments["steps_per_cycle"],
    )
    replay = replay_n1_collocation_boundary(triplet.current)
    shell = build_n1_actual_shell_kinematics(triplet)
    incident = evaluate_n1_incident_velocity(
        triplet.current, shell.closed_shell.mesh.centroids
    )
    source = solve_conditioned_neumann_source(
        shell.closed_shell.mesh,
        incident_velocity=incident.production_total,
        wall_velocity=shell.face_wall_velocity,
    )
    flux_ledger = closed_surface_flux_ledger(
        shell.closed_shell.mesh,
        incident,
        wall_velocity=shell.face_wall_velocity,
    )
    filament_topology = audit_n1_filament_shell_topology(
        triplet.current, shell
    )
    current_area_normal = (
        shell.closed_shell.mesh.areas[:, None]
        * shell.closed_shell.mesh.normals
    )
    wall_flux = float(np.sum(
        shell.face_wall_velocity * current_area_normal
    ))
    incident_flux = float(np.sum(
        incident.production_total * current_area_normal
    ))
    prescribed_neumann_flux = float(np.dot(
        source.right_hand_side, shell.closed_shell.mesh.areas
    ))
    topology = trailing_edge_topology_residual(
        triplet.current, shell, dt=triplet.dt
    )
    direction = trailing_edge_direction_residual(shell, source)
    after_hash = _hash_frame_arrays(frames)
    image_delta = (
        incident.physical_symmetry_candidate_total
        - incident.production_total
    )
    wake_limit = 3.0e-5
    residual_limit = 8.0e-5
    shell_refinement = []
    for shell_nc, shell_ns in (
        (4, 8), (6, 12), (8, 16), (10, 20), (12, 24)
    ):
        reference = rg.robowing_real(
            shell_nc,
            shell_ns,
            0.80,
            root_off=0.0,
            cosine_chord=True,
        )
        common = dict(
            A_f=np.radians(arguments["flap_amp_deg"]),
            A_t=np.radians(arguments["twist_amp_deg"]),
            Om=2.0 * np.pi * arguments["freq"],
            phi=np.radians(arguments["twist_phase_deg"]),
            x_ea=0.25 * 0.287,
            span=0.80,
            swept_axis=False,
            root_off=0.0,
            ramp=None,
        )
        previous_mean = twisted_corners(
            reference, triplet.current.time - triplet.dt, **common
        )
        current_mean = twisted_corners(
            reference, triplet.current.time, **common
        )
        next_mean = twisted_corners(
            reference, triplet.current.time + triplet.dt, **common
        )
        refined_shell = build_actual_shell_kinematics_from_mean_surfaces(
            previous_mean_surface=previous_mean,
            current_mean_surface=current_mean,
            next_mean_surface=next_mean,
            dt=triplet.dt,
        )
        refined_incident = evaluate_n1_incident_velocity(
            triplet.current,
            refined_shell.closed_shell.mesh.centroids,
        )
        refined_source = solve_conditioned_neumann_source(
            refined_shell.closed_shell.mesh,
            incident_velocity=refined_incident.production_total,
            wall_velocity=refined_shell.face_wall_velocity,
        )
        refined_flux_ledger = closed_surface_flux_ledger(
            refined_shell.closed_shell.mesh,
            refined_incident,
            wall_velocity=refined_shell.face_wall_velocity,
        )
        refined_area_normal = (
            refined_shell.closed_shell.mesh.areas[:, None]
            * refined_shell.closed_shell.mesh.normals
        )
        refined_wall_flux = float(np.sum(
            refined_shell.face_wall_velocity * refined_area_normal
        ))
        refined_incident_flux = float(np.sum(
            refined_incident.production_total * refined_area_normal
        ))
        shell_refinement.append(
            {
                "nc": shell_nc,
                "ns": shell_ns,
                "panel_count": len(refined_shell.closed_shell.mesh.faces),
                "relative_source_flux":
                    refined_source.relative_source_flux,
                "source_flux": refined_source.source_flux,
                "source_flux_l1_scale": (
                    abs(refined_source.source_flux)
                    / max(
                        refined_source.relative_source_flux,
                        np.finfo(float).tiny,
                    )
                ),
                "wall_volume_flux": refined_wall_flux,
                "incident_closed_surface_flux": refined_incident_flux,
                "prescribed_neumann_flux": float(np.dot(
                    refined_source.right_hand_side,
                    refined_shell.closed_shell.mesh.areas,
                )),
                "relative_no_penetration":
                    refined_source.relative_no_penetration_residual,
                "condition_number": refined_source.condition_number,
                "closed_surface_flux_ledger":
                    _flux_ledger_dict(refined_flux_ledger),
            }
        )
    topology_channels = {
        name: _filament_channel_dict(getattr(filament_topology, name))
        for name in (
            "bound_direct",
            "bound_image",
            "wake_direct",
            "wake_image_candidate",
        )
    }
    production_piercing_count = sum(
        topology_channels[name]["proper_shell_piercing_segment_count"]
        for name in ("bound_direct", "bound_image", "wake_direct")
    )
    return {
        "configuration": arguments,
        "frame_count": len(frames),
        "snapshot_time": triplet.current.time,
        "snapshot_dt": triplet.dt,
        "snapshot_phase": triplet.current.snapshot_phase,
        "wake_ring_count": len(triplet.current.wake_rings),
        "wake_arithmetic": triplet.current.wake_arithmetic,
        "wake_kernel_kind": triplet.current.wake_kernel_kind,
        "captured_wake_velocity_relative_error":
            replay.relative_captured_wake_velocity_error,
        "collocation_normal_residual_relative":
            replay.relative_normal_residual,
        "captured_wake_gate":
            replay.relative_captured_wake_velocity_error <= wake_limit,
        "collocation_replay_gate":
            replay.relative_normal_residual <= residual_limit,
        "closed_shell_vertex_count": len(shell.closed_shell.mesh.vertices),
        "closed_shell_panel_count": len(shell.closed_shell.mesh.faces),
        "closed_shell_boundary_edge_count":
            shell.closed_shell.mesh.boundary_edge_count,
        "closed_shell_nonmanifold_edge_count":
            shell.closed_shell.mesh.nonmanifold_edge_count,
        "closed_shell_orientation_mismatch_count":
            shell.closed_shell.mesh.orientation_mismatch_count,
        "mean_surface_change": shell.maximum_mean_surface_change,
        "material_pairing_error": shell.maximum_material_pairing_error,
        "wall_velocity_node_mapping_error":
            shell.maximum_wall_velocity_node_mapping_error,
        "conditioned_source_no_penetration_relative":
            source.relative_no_penetration_residual,
        "conditioned_source_flux_relative": source.relative_source_flux,
        "conditioned_source_flux": source.source_flux,
        "conditioned_source_flux_l1_scale": (
            abs(source.source_flux)
            / max(source.relative_source_flux, np.finfo(float).tiny)
        ),
        "wall_volume_flux": wall_flux,
        "incident_closed_surface_flux": incident_flux,
        "prescribed_neumann_flux": prescribed_neumann_flux,
        "closed_surface_flux_ledger": _flux_ledger_dict(flux_ledger),
        "shell_refinement": shell_refinement,
        "filament_shell_topology": topology_channels,
        "production_active_filament_piercing_count":
            production_piercing_count,
        "filament_shell_representation_compatible":
            production_piercing_count == 0,
        "te_base_thickness_over_chord_min": float(
            np.min(topology.base_thickness_over_chord)
        ),
        "te_base_thickness_over_chord_max": float(
            np.max(topology.base_thickness_over_chord)
        ),
        "n1_rear_line_offset_over_chord_min": float(
            np.min(topology.n1_rear_line_offset_over_chord)
        ),
        "n1_rear_line_offset_over_chord_max": float(
            np.max(topology.n1_rear_line_offset_over_chord)
        ),
        "n1_rear_line_kernel_identity_error":
            topology.rear_line_kernel_identity_error,
        "base_to_n1_rear_offset_ratio_min": float(
            np.min(topology.base_to_rear_offset_ratio)
        ),
        "base_to_n1_rear_offset_ratio_max": float(
            np.max(topology.base_to_rear_offset_ratio)
        ),
        "first_wake_step_over_chord_min": float(
            np.min(topology.first_wake_step_over_chord)
        ),
        "first_wake_step_over_chord_max": float(
            np.max(topology.first_wake_step_over_chord)
        ),
        "te_upper_tangent_normal_velocity_rms": float(
            np.sqrt(np.mean(
                direction.upper_tangent_normal_velocity**2
            ))
        ),
        "te_lower_tangent_normal_velocity_rms": float(
            np.sqrt(np.mean(
                direction.lower_tangent_normal_velocity**2
            ))
        ),
        "te_bisector_normal_velocity_rms": float(
            np.sqrt(np.mean(direction.bisector_normal_velocity**2))
        ),
        "te_pressure_residual_available":
            direction.pressure_residual_available,
        "te_pressure_residual_blocker":
            direction.pressure_residual_blocker,
        "physical_symmetry_candidate_delta_rms": float(
            np.sqrt(np.mean(np.sum(image_delta * image_delta, axis=1)))
        ),
        "production_identity_excludes_mirrored_wake": True,
        "input_hash_before": before_hash,
        "input_hash_after": after_hash,
        "input_immutability_gate": before_hash == after_hash,
        "numeric_result_bitwise_unchanged": numerical_unchanged,
        "numeric_result_mismatch_keys": numerical_failures,
        "L_wind_N": float(observed["L_wind"]),
        "T_wind_N": float(observed["T_wind"]),
    }


def main() -> dict:
    wp.init()
    manufactured = _manufactured_oracle()
    representative = _representative_case()
    shell_gate = (
        representative["closed_shell_boundary_edge_count"] == 0
        and representative["closed_shell_nonmanifold_edge_count"] == 0
        and representative["closed_shell_orientation_mismatch_count"] == 0
        and representative["mean_surface_change"] == 0.0
        and representative["material_pairing_error"] <= 5.0e-15
        and representative["wall_velocity_node_mapping_error"] <= 2.0e-12
    )
    passed = (
        manufactured["fp64_gate"]
        and manufactured["fp32_gate"]
        and representative["captured_wake_gate"]
        and representative["collocation_replay_gate"]
        and shell_gate
        and representative["input_immutability_gate"]
        and representative["numeric_result_bitwise_unchanged"]
        and representative["production_identity_excludes_mirrored_wake"]
    )
    result = {
        "artifact": "n1_to_actual_thickness_shell_read_only_adapter",
        "claim_node": "N3.1j3b6c",
        "implementation_role": "diagnostic_shadow",
        "manufactured_ring_oracle": manufactured,
        "representative_v41": representative,
        "actual_shell_kinematics_gate": shell_gate,
        "adapter_gate": "GO" if passed else "NO-GO",
        "conditioned_thick_shell_route_gate": (
            "GO"
            if representative["filament_shell_representation_compatible"]
            else "NO-GO"
        ),
        "conditioned_thick_shell_route_diagnosis": (
            "no active production filament pierces the actual shell"
            if representative["filament_shell_representation_compatible"]
            else "active frozen-N1 production filament pierces the actual "
                 "shell; source conditioning is representation-incompatible"
        ),
        "thick_body_kutta_qualified": False,
        "unified_pressure_qualified": False,
        "production_activation_allowed": False,
    }
    output = (
        HERE / "docs" / "diag"
        / "n1_thick_shell_adapter_results.json"
    )
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit("N1 thick-shell adapter guard failed")
    return result


if __name__ == "__main__":
    main()
