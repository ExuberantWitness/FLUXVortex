import sys
import unittest
from pathlib import Path

import numpy as np

PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from claim_runtime import ClaimGraph, RunConfig  # noqa: E402
from claim_runtime.distributed_doublet import (  # noqa: E402
    QuadraticDoubletAssembly,
    QuadraticDoubletPatch,
    QuadraticDoubletSurface,
)
from claim_runtime.work_conjugate_transfer import (  # noqa: E402
    rigid_body_jacobian,
)


NODES = PLATFORM / "claim_nodes"


def surface():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    # The only non-zero degree of freedom is the shared diagonal midside.
    # Hence the global P2 trace is continuous internally and exactly zero on
    # every unmatched physical boundary.
    face_mu = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.4],
            [0.0, 0.0, 0.0, 0.4, 0.0, 0.0],
        ]
    )
    return QuadraticDoubletSurface(vertices, faces, face_mu)


def two_patch_assembly():
    first = QuadraticDoubletSurface(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        [[0, 1, 2]],
        [[0, 0, 0, 0, 0.4, 0]],
    )
    second = QuadraticDoubletSurface(
        [[1, 0, 0], [1, 1, 0], [0, 1, 0]],
        [[0, 1, 2]],
        [[0, 0, 0, 0, 0, 0.4]],
    )
    assembly = QuadraticDoubletAssembly(
        [
            QuadraticDoubletPatch(
                "first",
                first,
                {
                    (0, 1): "zero",
                    (0, 2): "zero",
                    (1, 2): "interface:diagonal",
                },
            ),
            QuadraticDoubletPatch(
                "second",
                second,
                {
                    (0, 1): "zero",
                    (1, 2): "zero",
                    (0, 2): "interface:diagonal",
                },
            ),
        ]
    )
    return first, assembly


class SpatialShadowGraphTests(unittest.TestCase):
    def graph(self):
        return ClaimGraph.from_yaml(
            NODES,
            RunConfig(
                closure="n3_spatial_shadow",
                values={},
                sources={},
            ),
        )

    def test_nested_claims_are_the_executable_shadow_topology(self):
        graph = self.graph()
        self.assertEqual(
            graph.topology,
            ["N3.1j4", "N3.1j3", "N3.1i2a"],
        )
        manifest = graph.manifest().to_dict()
        self.assertEqual(manifest["closure"], "n3_spatial_shadow")
        self.assertEqual(
            [node["state"] for node in manifest["nodes"]],
            ["partial", "open", "validated"],
        )

    def test_spatial_failure_blocks_pressure_and_transfer(self):
        graph = self.graph()
        state = surface()
        context = graph.step(
            0,
            0.0,
            {
                "spatial_shadow_inputs": {
                    "surface": state,
                    "field_points": np.array([[0.3, 0.2, 0.8]]),
                    # Canonical Case1/2 has not passed; no pressure inputs are
                    # allowed to bypass this gate.
                    "canonical_field_gate_passed": False,
                }
            },
        )
        self.assertFalse(
            context.values["spatial_vortex_state"]["eligible_for_pressure"]
        )
        self.assertEqual(
            context.values["panel_pressure_state"]["status"],
            "blocked",
        )
        self.assertEqual(
            context.values["structural_load_state"]["status"],
            "blocked",
        )
        self.assertEqual(len(context.ledger.items), 0)

    def test_synthetic_pass_exercises_pressure_to_structure_without_force_booking(self):
        graph = self.graph()
        state = surface()
        point_position = np.array(
            [
                [2.0 / 3.0, 1.0 / 3.0, 0.0],
                [1.0 / 3.0, 2.0 / 3.0, 0.0],
            ]
        )
        jacobian = rigid_body_jacobian(point_position)
        context = graph.step(
            3,
            0.03,
            {
                "spatial_shadow_inputs": {
                    "surface": state,
                    "previous_surface": state,
                    "field_points": np.array(
                        [[0.2, 0.3, 0.8], [1.4, 0.5, -0.6]]
                    ),
                    "canonical_field_gate_passed": True,
                    "pressure": {
                        "density": 1.225,
                        "local_velocity": np.zeros((2, 3)),
                        "surface_gradient": np.zeros((2, 3)),
                        "potential_rate_channels": {
                            "bound_unsteady": np.array([0.8, -0.3])
                        },
                        "area": np.array([0.5, 0.5]),
                        "normal": np.tile([0.0, 0.0, 1.0], (2, 1)),
                    },
                    "structure": {
                        "point_position": point_position,
                        "kinematic_jacobian": jacobian,
                        "virtual_work_probes": np.eye(6),
                    },
                }
            },
        )
        self.assertTrue(
            context.values["spatial_vortex_state"]["eligible_for_pressure"]
        )
        self.assertEqual(
            context.values["panel_pressure_state"]["status"],
            "shadow",
        )
        structural = context.values["structural_load_state"]
        self.assertEqual(structural["status"], "shadow")
        self.assertTrue(structural["guards"]["virtual_work"].passed)
        self.assertTrue(structural["guards"]["rigid_resultant"].passed)
        # A diagnostic shadow never books production force.
        self.assertEqual(len(context.ledger.items), 0)

    def test_assembly_vertex_star_runs_but_cannot_bypass_canonical_gate(self):
        graph = self.graph()
        first, assembly = two_patch_assembly()
        context = graph.step(
            0,
            0.0,
            {
                "spatial_shadow_inputs": {
                    "surface": first,
                    "assembly": assembly,
                    "field_points": np.array([[0.3, 0.2, 0.8]]),
                    "canonical_field_gate_passed": False,
                }
            },
        )
        spatial = context.values["spatial_vortex_state"]
        projection = spatial["geometry_velocity"]
        self.assertIsNotNone(projection)
        self.assertTrue(projection.report.full_rank)
        self.assertEqual(
            projection.report.gauge,
            "sheet_average_normal_vertex_star_assembly",
        )
        self.assertFalse(spatial["eligible_for_pressure"])
        self.assertEqual(
            context.values["panel_pressure_state"]["status"],
            "blocked",
        )


if __name__ == "__main__":
    unittest.main()
