import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from claim_runtime.near_wall_field_contract import (  # noqa: E402
    NearWallFieldContractError,
    NearWallFieldDataset,
    NearWallFieldProvenance,
    RequiredNearWallCoverage,
    assert_no_forbidden_targets,
    validate_near_wall_field_dataset,
)


def manufactured_dataset(*, provenance=None):
    time = np.array([0.0, 0.1, 0.2])
    chart = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    nt, ns, nn = len(time), len(chart), 9

    wall_position = np.empty((nt, ns, 3))
    wall_velocity = np.empty_like(wall_position)
    normal = np.zeros_like(wall_position)
    tangent = np.zeros((nt, ns, 2, 3))
    coordinate = np.empty((nt, ns, nn))
    density = np.empty((nt, ns, nn))
    velocity = np.empty((nt, ns, nn, 3))
    edge_position = np.empty_like(wall_position)
    edge_velocity = np.empty_like(wall_position)

    for it, current_time in enumerate(time):
        displacement = 0.01*np.sin(2.0*np.pi*current_time)
        speed = 0.02*np.pi*np.cos(2.0*np.pi*current_time)
        for node, (x_coord, y_coord) in enumerate(chart):
            wall_position[it, node] = [x_coord, y_coord, displacement]
            wall_velocity[it, node] = [0.0, 0.0, speed]
            normal[it, node] = [0.0, 0.0, 1.0]
            tangent[it, node, 0] = [1.0, 0.0, 0.0]
            tangent[it, node, 1] = [0.0, 1.0, 0.0]
            thickness = 0.02+0.001*node
            coordinate[it, node] = np.linspace(0.0, thickness, nn)
            eta = coordinate[it, node]/thickness
            outer = np.array([
                7.0+0.2*node,
                0.6*np.sin(2.0*np.pi*current_time)+0.05*node,
                speed,
            ])
            velocity[it, node] = (
                wall_velocity[it, node, None, :]
                +eta[:, None]*(outer-wall_velocity[it, node])[None, :]
            )
            edge_velocity[it, node] = outer
            edge_position[it, node] = (
                wall_position[it, node]+thickness*normal[it, node]
            )
            density[it, node] = 1.18+0.01*eta

    if provenance is None:
        provenance = NearWallFieldProvenance(
            source_kind="manufactured",
            reference="analytic moving crossflow schema case",
            case_id="manufactured-moving-crossflow",
            split_role="test",
            independent_audit_id="",
            edge_convention="manufactured-last-sample",
            reynolds_min=1.1e5,
            reynolds_max=1.9e5,
            regime_flags=frozenset({
                "dynamic",
                "three_dimensional",
                "crossflow",
                "moving_wall",
            }),
        )
    return NearWallFieldDataset(
        provenance=provenance,
        time=time,
        surface_chart=chart,
        surface_faces=faces,
        side=np.array([1, 1, -1, -1]),
        normal_coordinate=coordinate,
        density=density,
        velocity=velocity,
        wall_position=wall_position,
        wall_velocity=wall_velocity,
        surface_normal=normal,
        tangent_basis=tangent,
        edge_position=edge_position,
        edge_velocity=edge_velocity,
        extraction_tolerance=1.0e-11,
    )


def target_coverage():
    return RequiredNearWallCoverage(
        reynolds_min=1.1e5,
        reynolds_max=1.9e5,
        required_regimes=frozenset({
            "dynamic",
            "three_dimensional",
            "crossflow",
            "transitional",
            "separated",
            "moving_wall",
        }),
        required_split_role="test",
    )


def rotation_matrix():
    axis = np.array([0.3, -0.4, 0.8])
    axis /= np.linalg.norm(axis)
    angle = 0.71
    cross = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    return (
        np.eye(3)*np.cos(angle)
        +(1.0-np.cos(angle))*np.outer(axis, axis)
        +np.sin(angle)*cross
    )


class NearWallFieldContractTests(unittest.TestCase):
    def test_valid_manufactured_identity_but_not_physical_promotion(self):
        report = validate_near_wall_field_dataset(
            manufactured_dataset(),
            requirements=target_coverage(),
        )
        self.assertTrue(report.schema_valid)
        self.assertTrue(report.identity_valid)
        self.assertFalse(report.coverage_complete)
        self.assertFalse(report.production_evidence_eligible)
        self.assertIn(
            "independent_non_manufactured_source",
            report.missing_coverage,
        )
        self.assertIn("independent_audit_id", report.missing_coverage)
        self.assertIn("regime:separated", report.missing_coverage)
        self.assertIn("regime:transitional", report.missing_coverage)

    def test_rotation_preserves_identity_residuals(self):
        original = manufactured_dataset()
        reference = validate_near_wall_field_dataset(original)
        rotation = rotation_matrix()
        rotated = replace(
            original,
            velocity=original.velocity@rotation.T,
            wall_position=original.wall_position@rotation.T,
            wall_velocity=original.wall_velocity@rotation.T,
            surface_normal=original.surface_normal@rotation.T,
            tangent_basis=original.tangent_basis@rotation.T,
            edge_position=original.edge_position@rotation.T,
            edge_velocity=original.edge_velocity@rotation.T,
        )
        transformed = validate_near_wall_field_dataset(rotated)
        self.assertTrue(transformed.identity_valid)
        np.testing.assert_allclose(
            [
                transformed.max_no_slip_error,
                transformed.max_edge_velocity_error,
                transformed.max_edge_position_error,
                transformed.max_normal_norm_error,
                transformed.max_basis_norm_error,
                transformed.max_basis_orthogonality_error,
                transformed.max_handedness_error,
            ],
            [
                reference.max_no_slip_error,
                reference.max_edge_velocity_error,
                reference.max_edge_position_error,
                reference.max_normal_norm_error,
                reference.max_basis_norm_error,
                reference.max_basis_orthogonality_error,
                reference.max_handedness_error,
            ],
            atol=5.0e-15,
        )

    def test_no_slip_violation_is_visible(self):
        original = manufactured_dataset()
        velocity = original.velocity.copy()
        velocity[1, 2, 0, 0] += 0.02
        report = validate_near_wall_field_dataset(
            replace(original, velocity=velocity)
        )
        self.assertFalse(report.identity_valid)
        self.assertGreater(report.max_no_slip_error, 0.019)
        self.assertFalse(report.production_evidence_eligible)

    def test_forbidden_target_rejected_before_construction(self):
        with self.assertRaisesRegex(
            NearWallFieldContractError,
            "forbidden target channel",
        ):
            assert_no_forbidden_targets({
                "profiles": {"velocity": [0.0, 1.0]},
                "targets": {"lift_target": [7.8]},
            })

    def test_malformed_surface_topology_fails(self):
        original = manufactured_dataset()
        with self.assertRaisesRegex(
            NearWallFieldContractError,
            "reference valid nodes",
        ):
            validate_near_wall_field_dataset(
                replace(
                    original,
                    surface_faces=np.array([[0, 1, 7]]),
                )
            )


if __name__ == "__main__":
    unittest.main()

