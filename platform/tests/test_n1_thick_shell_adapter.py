import sys
import unittest
from pathlib import Path

import numpy as np

PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from claim_runtime.hirato_shadow import mirrored_ring_field  # noqa: E402
from claim_runtime.n1_thick_shell_adapter import (  # noqa: E402
    N1StepSnapshot,
    N1SymmetryRole,
    N1ThickShellAdapterError,
    build_n1_actual_shell_kinematics,
    evaluate_n1_incident_velocity,
    parse_n1_snapshot_triplet,
    production_ring_field_velocity,
)
from claim_runtime.viscous_shell_geometry import (  # noqa: E402
    naca4_dual_surface_shell,
)
from diff_uvlm_unsteady import _ring_vel  # noqa: E402


def _frame(time, *, symmetry=True):
    nc = 2
    ns = 2
    xi = np.array([0.0, 0.5, 1.0])
    eta = np.array([0.0, 0.4, 0.8])
    reference = naca4_dual_surface_shell(
        xi,
        eta,
        np.full_like(eta, 0.3),
    )
    translation_velocity = np.array([0.1, -0.2, 0.3])
    corners = (
        reference.mean_surface
        + float(time) * translation_velocity
    )
    panel_count = nc * ns
    ring = np.array(
        [
            [0.05, 0.05, 0.02],
            [0.05, 0.35, 0.02],
            [0.20, 0.35, 0.02],
            [0.20, 0.05, 0.02],
        ]
    )
    bound = np.stack(
        [ring + np.array([0.02 * index, 0.0, 0.01 * index])
         for index in range(panel_count)]
    )
    collocation = np.array(
        [
            [0.12, 0.12, 0.09],
            [0.12, 0.55, 0.09],
            [0.25, 0.12, 0.09],
            [0.25, 0.55, 0.09],
        ]
    )
    return {
        "t": float(time),
        "bound": bound,
        "gam": np.linspace(0.1, 0.4, panel_count),
        "corners": corners,
        "corner_velocity": np.broadcast_to(
            translation_velocity, corners.shape
        ).copy(),
        "collocation_points": collocation,
        "panel_normals": np.broadcast_to(
            np.array([0.0, 0.0, 1.0]), collocation.shape
        ).copy(),
        "collocation_velocity": np.broadcast_to(
            translation_velocity, collocation.shape
        ).copy(),
        "wake_induced_velocity": np.zeros_like(collocation),
        "freestream_velocity": np.array([1.0, 0.0, 0.0]),
        "symmetry_enabled": bool(symmetry),
        "snapshot_phase": "post_force_pre_shed",
        "bound_kernel_kind": "singular",
        "bound_arithmetic": "fp64",
        "bound_denominator_floor": 1.0e-10,
        "wake_kernel_kind": "singular",
        "wake_arithmetic": "fp64",
        "wake_denominator_floor": 1.0e-10,
        "wake_core_delta": np.zeros(0),
        "wr": np.zeros((0, 4, 3)),
        "wg": np.zeros(0),
        "wtype": np.zeros(0, dtype=int),
        "nc": nc,
        "ns": ns,
    }


class N1ThickShellAdapterTests(unittest.TestCase):
    def test_fp64_ring_field_matches_frozen_numpy_oracle(self):
        ring = np.array(
            [[
                [0.0, 0.2, 0.0],
                [0.0, 1.0, 0.0],
                [0.7, 1.0, 0.1],
                [0.7, 0.2, 0.1],
            ]]
        )
        gamma = np.array([0.37])
        points = np.array(
            [[0.2, 0.4, 0.8], [1.8, 0.7, 1.2], [-0.6, 0.3, 0.5]]
        )
        result = production_ring_field_velocity(
            points,
            ring,
            gamma,
            arithmetic="fp64",
            denominator_floor=1.0e-10,
        )
        expected = np.asarray(
            [gamma[0] * _ring_vel(point, ring[0]) for point in points]
        )
        np.testing.assert_allclose(
            result, expected, rtol=4.0e-15, atol=4.0e-15
        )

    def test_production_and_candidate_wake_symmetry_are_separate(self):
        frame = _frame(0.0, symmetry=True)
        wake = np.array(
            [[
                [0.4, 0.1, 0.0],
                [0.4, 0.5, 0.0],
                [0.7, 0.5, 0.0],
                [0.7, 0.1, 0.0],
            ]]
        )
        frame["wr"] = wake
        frame["wg"] = np.array([0.2])
        frame["wtype"] = np.array([0])
        frame["wake_core_delta"] = np.zeros(1)
        snapshot = N1StepSnapshot.from_frame(frame)
        points = np.array([[0.3, 0.35, 0.4]])
        ledger = evaluate_n1_incident_velocity(snapshot, points)
        image = production_ring_field_velocity(
            points,
            mirrored_ring_field(wake),
            frame["wg"],
            arithmetic="fp64",
            denominator_floor=1.0e-10,
        )
        np.testing.assert_allclose(
            ledger.physical_symmetry_candidate_total
            - ledger.production_total,
            image,
        )
        np.testing.assert_array_equal(
            ledger.selected(N1SymmetryRole.PRODUCTION_IDENTITY),
            ledger.production_total,
        )

    def test_unknown_or_inferred_kernel_identity_fails(self):
        frame = _frame(0.0)
        del frame["wake_arithmetic"]
        with self.assertRaises(N1ThickShellAdapterError):
            N1StepSnapshot.from_frame(frame)
        frame = _frame(0.0)
        frame["wake_kernel_kind"] = "singular"
        frame["wr"] = np.zeros((1, 4, 3))
        frame["wg"] = np.ones(1)
        frame["wtype"] = np.zeros(1, dtype=int)
        frame["wake_core_delta"] = np.ones(1) * 0.05
        with self.assertRaises(N1ThickShellAdapterError):
            N1StepSnapshot.from_frame(frame)

    def test_triplet_builds_closed_objective_shell_without_mutation(self):
        frames = [_frame(-0.01), _frame(0.0), _frame(0.01)]
        before = [
            {key: value.copy() if isinstance(value, np.ndarray) else value
             for key, value in frame.items()}
            for frame in frames
        ]
        triplet = parse_n1_snapshot_triplet(
            frames, expected_dt=0.01
        )
        shell = build_n1_actual_shell_kinematics(triplet)
        self.assertEqual(shell.closed_shell.mesh.boundary_edge_count, 0)
        self.assertEqual(shell.closed_shell.mesh.nonmanifold_edge_count, 0)
        self.assertEqual(
            shell.closed_shell.mesh.orientation_mismatch_count, 0
        )
        self.assertEqual(shell.maximum_mean_surface_change, 0.0)
        self.assertLessEqual(
            shell.maximum_material_pairing_error, 1.0e-16
        )
        self.assertLessEqual(
            shell.maximum_wall_velocity_node_mapping_error, 2.0e-13
        )
        np.testing.assert_allclose(
            shell.face_wall_velocity,
            np.broadcast_to(
                np.array([0.1, -0.2, 0.3]),
                shell.face_wall_velocity.shape,
            ),
            rtol=0.0,
            atol=2.0e-13,
        )
        for source, original in zip(frames, before, strict=True):
            for key in source:
                if isinstance(source[key], np.ndarray):
                    np.testing.assert_array_equal(
                        source[key], original[key]
                    )
                else:
                    self.assertEqual(source[key], original[key])

    def test_nonconsecutive_or_skipped_triplet_fails(self):
        with self.assertRaises(N1ThickShellAdapterError):
            parse_n1_snapshot_triplet(
                [_frame(0.0), _frame(0.01), _frame(0.03)]
            )


if __name__ == "__main__":
    unittest.main()
