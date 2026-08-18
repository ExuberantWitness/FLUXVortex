import sys
import unittest
from pathlib import Path

import numpy as np

PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from claim_runtime.hirato_equations import (  # noqa: E402
    HiratoEquationError,
    RAMESH_2014_LDVM_V25_BIRTH_SOURCE_TAG,
    cutoff_radius_family,
    embed_chord_normal_displacement,
    first_lev_displacement_ramesh_2d,
    first_lev_displacement_ramesh_2014_ldvm_v25,
    first_vortex_displacement_ansari,
    kelvin_residual_eq9,
    lesp_eq6,
    lesp_sensitivity_eq6,
    potential_rate_eq17,
    preconstraint_shed_mask,
    rearmost_bound_aft_edge,
    rollup_displacement_eq24,
    solve_lesp_constraint,
    tev_first_displacement_hirato,
    tev_strength_eq9,
)


class HiratoEquationTests(unittest.TestCase):
    def test_ansari_first_vortex_is_local_velocity_half_step(self):
        edge_velocity = np.array(
            [
                [[2.0, 0.0, 1.0], [1.0, -2.0, 0.5]],
                [[-1.0, 3.0, 0.0], [0.0, 4.0, -2.0]],
            ]
        )
        np.testing.assert_allclose(
            first_vortex_displacement_ansari(edge_velocity, dt=0.04),
            0.02 * edge_velocity,
        )

    def test_ramesh_first_lev_preserves_published_2d_components(self):
        a0 = np.array([0.27, -0.27])
        alpha = np.deg2rad(np.array([30.0, 30.0]))
        displacement = first_lev_displacement_ramesh_2d(
            u_infinity=4.0,
            a0=a0,
            alpha_rad=alpha,
            dt=0.05,
        )
        scale = 4.0 * a0 * 0.05 / np.sqrt(2.0)
        np.testing.assert_allclose(displacement[:, 0], scale * 0.5)
        np.testing.assert_allclose(
            displacement[:, 1],
            scale * np.cos(np.deg2rad(30.0)),
        )

    def test_ramesh_2014_ldvm_v25_first_or_restart_uses_local_half_step(self):
        edge_velocity = np.array(
            [
                [[2.0, -1.0, 0.5], [1.0, 3.0, -2.0]],
                [[-0.2, 0.4, 1.1], [4.0, -3.0, 2.0]],
            ]
        )
        displacement = first_lev_displacement_ramesh_2014_ldvm_v25(
            edge_velocity,
            dt=0.04,
        )
        np.testing.assert_array_equal(displacement, 0.02 * edge_velocity)
        self.assertIn("ramesh-jfm-2014", RAMESH_2014_LDVM_V25_BIRTH_SOURCE_TAG)
        self.assertIn("ldvm-v2.5", RAMESH_2014_LDVM_V25_BIRTH_SOURCE_TAG)

    def test_2d_to_finite_wing_embedding_requires_explicit_orthonormal_basis(self):
        displacement = np.array([[0.1, 0.2], [-0.3, 0.4]])
        tangent = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        normal = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
        embedded = embed_chord_normal_displacement(
            displacement,
            tangent,
            normal,
        )
        np.testing.assert_allclose(
            embedded,
            np.array([[0.1, 0.0, 0.2], [0.0, -0.3, -0.4]]),
        )
        with self.assertRaises(HiratoEquationError):
            embed_chord_normal_displacement(
                displacement,
                tangent,
                tangent,
            )

    def test_hirato_tev_uses_dissertation_0p3_translation(self):
        velocity = np.array([3.0, 0.0, -0.5])
        np.testing.assert_allclose(
            tev_first_displacement_hirato(velocity, dt=0.02),
            0.3 * velocity * 0.02,
        )

    def test_eq24_uses_current_previous_velocity_average(self):
        current = np.array([[2.0, 1.0, 0.0], [0.0, -1.0, 3.0]])
        previous = np.array([[1.0, 3.0, 2.0], [2.0, 1.0, 1.0]])
        np.testing.assert_allclose(
            rollup_displacement_eq24(current, previous, dt=0.1),
            0.05 * (current + previous),
        )

    def test_cutoff_family_covers_registered_range_without_default_pick(self):
        np.testing.assert_allclose(
            cutoff_radius_family(0.004),
            [0.0004, 0.001, 0.00196],
        )
        with self.assertRaises(HiratoEquationError):
            cutoff_radius_family(0.004, ratios=[0.1, 0.5])

    def test_eq6_pairs_gamma1_with_actual_forward_lattice_width(self):
        gamma = np.array([0.2, -0.1])
        chord = np.array([1.0, 0.5])
        delta_x = 0.25 * chord
        theta = np.arccos(0.5)
        expected = 1.13 * gamma / (3.0 * chord * (theta + np.sin(theta)))
        np.testing.assert_allclose(
            lesp_eq6(gamma, 3.0, chord, delta_x),
            expected,
        )
        fixed_xref = lesp_eq6(gamma, 3.0, chord, 0.10 * chord)
        self.assertGreater(abs(float(fixed_xref[0])), abs(float(expected[0])))

    def test_eq6_sensitivity_and_constraint_close_coupled_active_strips(self):
        ns, nc = 2, 3
        response = np.zeros((ns, nc * ns))
        response[:, :ns] = np.array([[-0.7, -0.1], [-0.2, -0.8]])
        chord = np.ones(ns)
        delta_x = np.full(ns, 0.25)
        sensitivity = lesp_sensitivity_eq6(
            response,
            u_infinity=3.0,
            chord=chord,
            delta_x_front=delta_x,
        )
        pre = np.array([0.34, -0.31])
        solution = solve_lesp_constraint(
            pre,
            sensitivity,
            active=np.array([True, True]),
            lesp_crit=0.27,
        )
        np.testing.assert_allclose(
            solution.a0_post,
            [0.27, -0.27],
            atol=1e-14,
        )
        np.testing.assert_allclose(solution.active_residual, 0.0, atol=1e-14)
        self.assertTrue(np.isfinite(solution.condition_number))

    def test_constraint_refuses_singular_active_system(self):
        with self.assertRaises(HiratoEquationError):
            solve_lesp_constraint(
                np.array([0.31, 0.32]),
                np.ones((2, 2)),
                active=np.array([True, True]),
                lesp_crit=0.27,
            )

    def test_pseudovortex_uses_rearmost_bound_aft_edge(self):
        nc, ns = 3, 2
        rings = np.zeros((nc * ns, 4, 3))
        for chord_row in range(nc):
            sl = slice(chord_row * ns, (chord_row + 1) * ns)
            rings[sl, 2, 0] = 10.0 + chord_row
            rings[sl, 3, 0] = 20.0 + chord_row
        aft_right, aft_left = rearmost_bound_aft_edge(rings, nc, ns)
        np.testing.assert_allclose(aft_right[:, 0], 12.0)
        np.testing.assert_allclose(aft_left[:, 0], 22.0)
        self.assertFalse(np.allclose(aft_right, rings[:ns, 2]))

    def test_eq9_kelvin_residual_closes_only_with_previous_lev(self):
        prev_bound = np.array([0.4, -0.2, 0.1])
        prev_lev = np.array([0.08, -0.03, 0.0])
        tev = tev_strength_eq9(prev_bound, prev_lev)
        np.testing.assert_allclose(
            kelvin_residual_eq9(tev, prev_bound, prev_lev),
            0.0,
            atol=0.0,
        )
        self.assertGreater(
            np.max(np.abs(kelvin_residual_eq9(prev_bound, prev_bound, prev_lev))),
            0.0,
        )

    def test_eq17_contains_stripwise_lev_rate(self):
        bound_prev = np.array([[0.1, 0.2], [0.3, 0.4]])
        bound_now = bound_prev + np.array([[0.01, 0.02], [0.03, 0.04]])
        rate = potential_rate_eq17(
            bound_now,
            bound_prev,
            lev_now=np.array([0.06, 0.08]),
            lev_prev=np.array([0.02, 0.03]),
            active=np.array([True, False]),
            dt=0.02,
        )
        np.testing.assert_allclose(rate.lev[:, 0], 2.0)
        np.testing.assert_allclose(rate.lev[:, 1], 0.0)
        np.testing.assert_allclose(rate.total, rate.bound + rate.lev)

    def test_shed_event_must_use_preconstraint_lesp(self):
        pre = np.array([0.31, -0.28, 0.20])
        post = np.array([0.27, -0.27, 0.20])
        np.testing.assert_array_equal(
            preconstraint_shed_mask(pre, 0.27),
            np.array([True, True, False]),
        )
        np.testing.assert_array_equal(
            preconstraint_shed_mask(post, 0.27),
            np.array([False, False, False]),
        )

    def test_invalid_shapes_fail_explicitly(self):
        with self.assertRaises(HiratoEquationError):
            rearmost_bound_aft_edge(np.zeros((2, 4, 3)), nc=2, ns=2)
        with self.assertRaises(HiratoEquationError):
            potential_rate_eq17(
                np.zeros((2, 2)),
                np.zeros((2, 3)),
                np.zeros(2),
                np.zeros(2),
                np.ones(2, dtype=bool),
                0.1,
            )


if __name__ == "__main__":
    unittest.main()
