"""Run the preregistered S3k moving-curved multi-patch transport gate."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from claim_runtime.distributed_doublet import (  # noqa: E402
    QuadraticDoubletAssembly,
    QuadraticDoubletPatch,
    QuadraticDoubletSurface,
)
from claim_runtime.p2_surface_material_transport import (  # noqa: E402
    assemble_p2_patch_material_transport,
    assemble_p2_surface_material_transport,
)


CASES = (
    HERE / "docs" / "diag"
    / "moving_p2_wake_transport_cases.yaml"
)
RESULTS = (
    HERE / "docs" / "diag"
    / "moving_p2_wake_transport_results.json"
)


def _rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    direction = axis / np.linalg.norm(axis)
    cross = np.array(
        (
            (0.0, -direction[2], direction[1]),
            (direction[2], 0.0, -direction[0]),
            (-direction[1], direction[0], 0.0),
        )
    )
    return (
        np.cos(angle) * np.eye(3)
        + (1.0 - np.cos(angle)) * np.outer(direction, direction)
        + np.sin(angle) * cross
    )


def _geometry(
    first: np.ndarray,
    second: np.ndarray,
    time: float,
    *,
    epsilon: float,
    omega: float,
) -> np.ndarray:
    return np.column_stack(
        (
            first,
            second,
            (
                epsilon
                * np.sin(np.pi * first)
                * np.sin(np.pi * second)
                * np.sin(omega * time)
            ),
        )
    )


def _mu0(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return first * (1.0 - first) * second * (1.0 - second)


def _flow_inverse(value: np.ndarray, coefficient: float, time: float) -> np.ndarray:
    return (
        2.0
        / np.pi
        * np.arctan(
            np.tan(0.5 * np.pi * value)
            * np.exp(-coefficient * np.pi * time)
        )
    )


def _exact_mu(
    first: np.ndarray,
    second: np.ndarray,
    time: float,
    *,
    coefficient_first: float,
    coefficient_second: float,
) -> np.ndarray:
    return _mu0(
        _flow_inverse(first, coefficient_first, time),
        _flow_inverse(second, coefficient_second, time),
    )


def _relative_velocity(
    points: np.ndarray,
    time: float,
    *,
    epsilon: float,
    omega: float,
    coefficient_first: float,
    coefficient_second: float,
) -> np.ndarray:
    first = points[:, 0]
    second = points[:, 1]
    slope_first = (
        epsilon
        * np.pi
        * np.cos(np.pi * first)
        * np.sin(np.pi * second)
        * np.sin(omega * time)
    )
    slope_second = (
        epsilon
        * np.pi
        * np.sin(np.pi * first)
        * np.cos(np.pi * second)
        * np.sin(omega * time)
    )
    rate_first = coefficient_first * np.sin(np.pi * first)
    rate_second = coefficient_second * np.sin(np.pi * second)
    return np.column_stack(
        (
            rate_first,
            rate_second,
            slope_first * rate_first + slope_second * rate_second,
        )
    )


def _patch_assembly(
    time: float,
    *,
    streamwise_cells: int,
    spanwise_cells: int,
    epsilon: float,
    omega: float,
    rotation: np.ndarray | None = None,
    translation: np.ndarray | None = None,
) -> QuadraticDoubletAssembly:
    span = np.linspace(0.0, 1.0, spanwise_cells + 1)
    patches = []
    for strip in range(streamwise_cells):
        first0 = strip / streamwise_cells
        first1 = (strip + 1) / streamwise_cells
        parameter_vertices = np.vstack(
            (
                np.column_stack(
                    (
                        np.full_like(span, first0),
                        span,
                    )
                ),
                np.column_stack(
                    (
                        np.full_like(span, first1),
                        span,
                    )
                ),
            )
        )
        vertices = _geometry(
            parameter_vertices[:, 0],
            parameter_vertices[:, 1],
            time,
            epsilon=epsilon,
            omega=omega,
        )
        if rotation is not None:
            vertices = vertices @ rotation.T + translation
        offset = spanwise_cells + 1
        faces = []
        for index in range(spanwise_cells):
            v00 = index
            v01 = index + 1
            v10 = offset + index
            v11 = offset + index + 1
            if (strip + index) % 2 == 0:
                faces.extend(
                    ((v00, v10, v11), (v00, v11, v01))
                )
            else:
                faces.extend(
                    ((v00, v10, v01), (v10, v11, v01))
                )
        face_array = np.asarray(faces, dtype=np.int64)
        face_mu = np.empty((len(face_array), 6), dtype=float)
        for face_index, face in enumerate(face_array):
            triangle = parameter_vertices[face]
            nodes = np.vstack(
                (
                    triangle,
                    0.5 * (triangle[0] + triangle[1]),
                    0.5 * (triangle[1] + triangle[2]),
                    0.5 * (triangle[2] + triangle[0]),
                )
            )
            face_mu[face_index] = _mu0(nodes[:, 0], nodes[:, 1])
        surface = QuadraticDoubletSurface(
            vertices,
            face_array,
            face_mu,
        )
        roles = {}
        for index in range(spanwise_cells):
            older = tuple(sorted((index, index + 1)))
            newer = tuple(
                sorted(
                    (
                        offset + index,
                        offset + index + 1,
                    )
                )
            )
            roles[older] = (
                "zero"
                if strip == 0
                else f"interface:time-{strip}:span-{index}"
            )
            roles[newer] = (
                "zero"
                if strip == streamwise_cells - 1
                else f"interface:time-{strip + 1}:span-{index}"
            )
        roles[tuple(sorted((0, offset)))] = "zero"
        roles[
            tuple(
                sorted(
                    (
                        spanwise_cells,
                        offset + spanwise_cells,
                    )
                )
            )
        ] = "zero"
        patches.append(
            QuadraticDoubletPatch(
                f"chronological-strip-{strip}",
                surface,
                roles,
            )
        )
    return QuadraticDoubletAssembly(patches)


def _operator(
    time: float,
    contract: dict,
    *,
    rotation: np.ndarray | None = None,
    translation: np.ndarray | None = None,
):
    canonical = contract["canonical"]
    cells = canonical["cells"]
    geometry = canonical["geometry"]
    rates = canonical["relative_material_velocity"]["parameter_rates"]
    epsilon = float(geometry["epsilon"])
    omega = float(geometry["omega"])
    coefficient_first = float(rates["a"])
    coefficient_second = float(rates["b"])
    assembly = _patch_assembly(
        time,
        streamwise_cells=int(cells["streamwise"]),
        spanwise_cells=int(cells["spanwise"]),
        epsilon=epsilon,
        omega=omega,
        rotation=rotation,
        translation=translation,
    )

    def velocity(points: np.ndarray) -> np.ndarray:
        if rotation is None:
            base = points
        else:
            base = (points - translation) @ rotation
        value = _relative_velocity(
            base,
            time,
            epsilon=epsilon,
            omega=omega,
            coefficient_first=coefficient_first,
            coefficient_second=coefficient_second,
        )
        return value if rotation is None else value @ rotation.T

    patch = assemble_p2_patch_material_transport(
        assembly,
        relative_velocity_provider=velocity,
        quadrature_order=int(canonical["quadrature_order"]),
    )
    topology = patch.operator.topology
    monolithic = assemble_p2_surface_material_transport(
        topology.vertices,
        topology.faces,
        relative_velocity_provider=velocity,
        quadrature_order=int(canonical["quadrature_order"]),
    )
    return assembly, patch, monolithic


def _integrate(
    contract: dict,
    *,
    steps: int,
    rotation: np.ndarray | None = None,
    translation: np.ndarray | None = None,
):
    canonical = contract["canonical"]
    start = float(canonical["time"]["start"])
    end = float(canonical["time"]["end"])
    dt = (end - start) / steps
    assembly0, patch0, monolithic0 = _operator(
        start,
        contract,
        rotation=rotation,
        translation=translation,
    )
    patch_state = patch0.extract_patch_scalar(assembly0)
    monolithic_state = patch_state.copy()
    matrix_difference = 0.0
    geometry_gap = 0.0
    constant_residual = 0.0
    minimum_area_ratio = np.inf
    initial_areas = np.array(
        [
            patch.surface.element(face).area
            for patch in assembly0.patches
            for face in range(len(patch.surface))
        ]
    )
    last_patch = patch0
    for step in range(steps):
        time = start + step * dt
        assembly, patch, monolithic = _operator(
            time,
            contract,
            rotation=rotation,
            translation=translation,
        )
        next_assembly, next_patch, next_monolithic = _operator(
            time + dt,
            contract,
            rotation=rotation,
            translation=translation,
        )
        matrix_difference = max(
            matrix_difference,
            float(
                np.max(
                    np.abs(
                        patch.operator.mass_matrix
                        - monolithic.mass_matrix
                    ),
                    initial=0.0,
                )
            ),
            float(
                np.max(
                    np.abs(
                        patch.operator.advection_matrix
                        - monolithic.advection_matrix
                    ),
                    initial=0.0,
                )
            ),
            float(
                np.max(
                    np.abs(
                        next_patch.operator.mass_matrix
                        - next_monolithic.mass_matrix
                    ),
                    initial=0.0,
                )
            ),
            float(
                np.max(
                    np.abs(
                        next_patch.operator.advection_matrix
                        - next_monolithic.advection_matrix
                    ),
                    initial=0.0,
                )
            ),
        )
        geometry_gap = max(
            geometry_gap,
            patch.maximum_interface_geometry_gap,
            next_patch.maximum_interface_geometry_gap,
        )
        constant_residual = max(
            constant_residual,
            patch.operator.constant_rate_residual,
            next_patch.operator.constant_rate_residual,
        )
        predictor = patch_state + dt * patch.rate(patch_state)
        patch_state = patch_state + 0.5 * dt * (
            patch.rate(patch_state) + next_patch.rate(predictor)
        )
        mono_predictor = (
            monolithic_state
            + dt * monolithic.rate(monolithic_state)
        )
        monolithic_state = monolithic_state + 0.5 * dt * (
            monolithic.rate(monolithic_state)
            + next_monolithic.rate(mono_predictor)
        )
        current_areas = np.array(
            [
                item.surface.element(face).area
                for item in next_assembly.patches
                for face in range(len(item.surface))
            ]
        )
        minimum_area_ratio = min(
            minimum_area_ratio,
            float(np.min(current_areas / initial_areas)),
        )
        last_patch = next_patch
    return {
        "patch_state": patch_state,
        "monolithic_state": monolithic_state,
        "patch_monolithic_matrix_difference": matrix_difference,
        "patch_monolithic_final_difference": float(
            np.max(
                np.abs(patch_state - monolithic_state),
                initial=0.0,
            )
        ),
        "geometry_gap": geometry_gap,
        "constant_residual": constant_residual,
        "minimum_area_ratio": minimum_area_ratio,
        "final_patch": last_patch,
    }


def _relative_mass_error(operator, value, exact) -> float:
    error = value - exact
    mass = operator.mass_matrix
    return float(
        np.sqrt(
            float(error @ mass @ error)
            / float(exact @ mass @ exact)
        )
    )


def _trace_and_boundary(
    patch,
    value: np.ndarray,
    *,
    rotation: np.ndarray | None = None,
    translation: np.ndarray | None = None,
) -> tuple[float, float]:
    local_values = patch.patch_face_values(value)
    members: dict[int, list[float]] = {}
    for dofs, face_values in zip(
        patch.patch_face_dofs,
        local_values,
    ):
        for dof, local in zip(dofs.ravel(), face_values.ravel()):
            members.setdefault(int(dof), []).append(float(local))
    trace = max(
        (max(values) - min(values) for values in members.values()),
        default=0.0,
    )
    coordinates = patch.operator.topology.degree_of_freedom_coordinates
    if rotation is not None:
        coordinates = (coordinates - translation) @ rotation
    boundary = (
        np.isclose(coordinates[:, 0], 0.0)
        | np.isclose(coordinates[:, 0], 1.0)
        | np.isclose(coordinates[:, 1], 0.0)
        | np.isclose(coordinates[:, 1], 1.0)
    )
    return trace, float(
        np.max(np.abs(value[boundary]), initial=0.0)
    )


def run() -> dict:
    contract = yaml.safe_load(CASES.read_text(encoding="utf-8"))
    canonical = contract["canonical"]
    thresholds = contract["thresholds"]
    steps = [int(value) for value in canonical["time"]["step_families"]]
    integrations = [
        _integrate(contract, steps=value)
        for value in steps
    ]
    states = [item["patch_state"] for item in integrations]
    coarse_change = float(
        np.max(np.abs(states[1] - states[0]), initial=0.0)
    )
    fine_change = float(
        np.max(np.abs(states[2] - states[1]), initial=0.0)
    )
    time_ratio = coarse_change / max(
        fine_change,
        np.finfo(float).tiny,
    )
    finest = integrations[-1]
    coordinates = (
        finest["final_patch"].operator.topology
        .degree_of_freedom_coordinates
    )
    rates = canonical["relative_material_velocity"]["parameter_rates"]
    exact = _exact_mu(
        coordinates[:, 0],
        coordinates[:, 1],
        float(canonical["time"]["end"]),
        coefficient_first=float(rates["a"]),
        coefficient_second=float(rates["b"]),
    )
    relative_error = _relative_mass_error(
        finest["final_patch"].operator,
        finest["patch_state"],
        exact,
    )
    trace, boundary = _trace_and_boundary(
        finest["final_patch"],
        finest["patch_state"],
    )

    rigid = canonical["rigid_frame_counterfactual"]
    rotation = _rotation_matrix(
        np.asarray(rigid["rotation_axis"], dtype=float),
        np.deg2rad(float(rigid["rotation_deg"])),
    )
    translation = np.asarray(rigid["translation"], dtype=float)
    moved = _integrate(
        contract,
        steps=steps[-1],
        rotation=rotation,
        translation=translation,
    )
    rigid_scalar = float(
        np.max(
            np.abs(moved["patch_state"] - finest["patch_state"]),
            initial=0.0,
        )
    )
    moved_coordinates = (
        moved["final_patch"].operator.topology
        .degree_of_freedom_coordinates
    )
    rigid_geometry = float(
        np.max(
            np.linalg.norm(
                moved_coordinates
                - (coordinates @ rotation.T + translation),
                axis=1,
            ),
            initial=0.0,
        )
    )
    max_matrix = max(
        item["patch_monolithic_matrix_difference"]
        for item in integrations + [moved]
    )
    max_final = max(
        item["patch_monolithic_final_difference"]
        for item in integrations + [moved]
    )
    geometry_gap = max(
        item["geometry_gap"] for item in integrations + [moved]
    )
    constant_residual = max(
        item["constant_residual"] for item in integrations + [moved]
    )
    minimum_area = min(
        item["minimum_area_ratio"] for item in integrations + [moved]
    )
    checks = {
        "explicit_patch_interfaces_are_welded": (
            geometry_gap
            <= float(thresholds["welded_geometry_gap_abs_max"])
        ),
        "patch_monolithic_matrices_match": (
            max_matrix
            <= float(
                thresholds[
                    "patch_monolithic_matrix_abs_difference_max"
                ]
            )
        ),
        "patch_monolithic_trajectories_match": (
            max_final
            <= float(
                thresholds[
                    "patch_monolithic_final_scalar_abs_difference_max"
                ]
            )
        ),
        "shared_P2_traces_are_exact": (
            trace
            <= float(thresholds["shared_trace_jump_abs_max"])
        ),
        "zero_boundaries_remain_zero": (
            boundary
            <= float(thresholds["zero_boundary_trace_abs_max"])
        ),
        "constant_scalar_is_instantaneously_preserved": (
            constant_residual
            <= float(thresholds["constant_rate_abs_max"])
        ),
        "Heun_time_Cauchy_gate_passes": (
            time_ratio
            >= float(thresholds["time_cauchy_ratio_min"])
        ),
        "finest_relative_L2_error_passes": (
            relative_error
            <= float(thresholds["finest_relative_l2_error_max"])
        ),
        "rigid_geometry_is_objective": (
            rigid_geometry
            <= float(
                thresholds["rigid_geometry_abs_difference_max"]
            )
        ),
        "rigid_scalar_is_objective": (
            rigid_scalar
            <= float(
                thresholds["rigid_final_scalar_abs_difference_max"]
            )
        ),
        "face_areas_remain_valid": (
            minimum_area
            >= float(thresholds["minimum_face_area_ratio_min"])
        ),
    }
    result = {
        "artifact": "moving_p2_wake_transport_oracle",
        "stage": contract["stage"],
        "claim_node": contract["claim_node"],
        "stage_decision": (
            "GO" if all(checks.values()) else "NO-GO"
        ),
        "checks": checks,
        "aggregate_metrics": {
            "welded_geometry_gap_abs_max": geometry_gap,
            "patch_monolithic_matrix_abs_difference_max": max_matrix,
            "patch_monolithic_final_scalar_abs_difference_max": max_final,
            "shared_trace_jump_abs_max": trace,
            "zero_boundary_trace_abs_max": boundary,
            "constant_rate_abs_max": constant_residual,
            "coarse_to_medium_scalar_abs_change": coarse_change,
            "medium_to_fine_scalar_abs_change": fine_change,
            "time_cauchy_ratio": time_ratio,
            "finest_relative_l2_error": relative_error,
            "rigid_geometry_abs_difference": rigid_geometry,
            "rigid_final_scalar_abs_difference": rigid_scalar,
            "minimum_face_area_ratio": minimum_area,
        },
        "forbidden_quantities_absent": [
            "proximity_welding",
            "seam_averaging",
            "mass_lumping",
            "upwind",
            "artificial_diffusion",
            "actual_induced_velocity",
            "pressure",
            "force",
            "target_load",
            "structural_dynamics",
        ],
        "production_activation_allowed": False,
    }
    RESULTS.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    payload = run()
    print(
        json.dumps(payload, ensure_ascii=False, indent=2),
        flush=True,
    )
    raise SystemExit(
        0 if payload["stage_decision"] == "GO" else 1
    )
