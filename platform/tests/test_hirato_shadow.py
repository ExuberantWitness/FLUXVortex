import sys
import unittest
from pathlib import Path

import numpy as np

PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from claim_runtime.hirato_equations import HiratoEquationError  # noqa: E402
from claim_runtime.hirato_shadow import HiratoSheetShadow  # noqa: E402


class HiratoSheetShadowTests(unittest.TestCase):
    def setUp(self):
        self.le = np.array(
            [
                [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[0.0, 1.0, 0.0], [0.0, 2.0, 0.0]],
            ]
        )
        self.first = self.le.copy()
        self.first[:, :, 0] = 0.3

    def test_first_ring_requires_explicit_placement(self):
        state = HiratoSheetShadow(ns=2)
        with self.assertRaises(HiratoEquationError):
            state.shed(
                step=0,
                leading_edges=self.le,
                gamma_now=np.array([0.2, 0.0]),
                active=np.array([True, False]),
            )

    def test_eq7_splits_nearest_ring_and_preserves_old_strength(self):
        state = HiratoSheetShadow(ns=2)
        state.shed(
            step=0,
            leading_edges=self.le,
            gamma_now=np.array([0.2, 0.0]),
            active=np.array([True, False]),
            first_aft_edges=self.first,
        )
        report = state.shed(
            step=1,
            leading_edges=self.le,
            gamma_now=np.array([0.35, 0.0]),
            active=np.array([True, False]),
        )
        self.assertEqual(len(state.rings), 2)
        np.testing.assert_allclose(state.gamma, [0.2, 0.35])
        expected_split = self.le[0].copy()
        expected_split[:, 0] = 0.1
        np.testing.assert_allclose(state.rings[0, :2], expected_split)
        np.testing.assert_allclose(state.rings[1, 3], expected_split[0])
        np.testing.assert_allclose(state.rings[1, 2], expected_split[1])
        np.testing.assert_allclose(report.split_residual, 0.0, atol=0.0)

    def test_created_strength_assignment_is_identity_checked(self):
        state = HiratoSheetShadow(ns=2)
        report = state.shed(
            step=0,
            leading_edges=self.le,
            gamma_now=np.zeros(2),
            active=np.array([True, True]),
            first_aft_edges=self.first,
        )
        state.assign_created_strengths(report.created_ids, [0.2, -0.1])
        np.testing.assert_allclose(state.gamma, [0.2, -0.1])
        with self.assertRaises(HiratoEquationError):
            state.assign_created_strengths([999], [1.0])
        with self.assertRaises(HiratoEquationError):
            state.assign_created_strengths(
                [report.created_ids[0], report.created_ids[0]],
                [1.0, 2.0],
            )

    def test_free_convection_uses_supplied_vertex_velocity(self):
        state = HiratoSheetShadow(ns=2)
        state.shed(
            step=0,
            leading_edges=self.le,
            gamma_now=np.array([0.2, -0.1]),
            active=np.array([True, True]),
            first_aft_edges=self.first,
        )
        before = state.rings.copy()
        velocity = np.zeros_like(before)
        velocity[..., 0] = 2.0
        velocity[..., 2] = -0.5
        state.convect(velocity, dt=0.25)
        np.testing.assert_allclose(state.rings[..., 0], before[..., 0] + 0.5)
        np.testing.assert_allclose(state.rings[..., 2], before[..., 2] - 0.125)

    def test_eq24_convects_birth_step_then_uses_velocity_history(self):
        state = HiratoSheetShadow(ns=2)
        state.shed(
            step=4,
            leading_edges=self.le,
            gamma_now=np.array([0.2, 0.0]),
            active=np.array([True, False]),
            first_aft_edges=self.first,
        )
        born = state.rings.copy()
        velocity_4 = np.full_like(state.rings, 2.0)
        report_4 = state.convect_eq24(velocity_4, dt=0.1, step=4)
        np.testing.assert_allclose(state.rings, born + 0.2)
        self.assertTrue(np.all(report_4.bootstrap_vertex))

        after_4 = state.rings.copy()
        velocity_5 = np.full_like(state.rings, 4.0)
        report_5 = state.convect_eq24(velocity_5, dt=0.1, step=5)
        np.testing.assert_allclose(state.rings, after_4 + 0.3)
        self.assertFalse(np.any(report_5.bootstrap_vertex))

    def test_eq7_remesh_bootstraps_only_new_material_vertices(self):
        state = HiratoSheetShadow(ns=2)
        state.shed(
            step=0,
            leading_edges=self.le,
            gamma_now=np.array([0.2, 0.0]),
            active=np.array([True, False]),
            first_aft_edges=self.first,
        )
        state.convect_eq24(np.full_like(state.rings, 1.0), dt=0.1, step=0)
        state.shed(
            step=1,
            leading_edges=self.le,
            gamma_now=np.array([0.3, 0.0]),
            active=np.array([True, False]),
        )
        report = state.convect_eq24(
            np.full_like(state.rings, 3.0),
            dt=0.1,
            step=1,
        )
        expected = np.array(
            [
                [True, True, False, False],
                [True, True, True, True],
            ]
        )
        np.testing.assert_array_equal(report.bootstrap_vertex, expected)
        np.testing.assert_allclose(report.displacement[0, :2], 0.3)
        np.testing.assert_allclose(report.displacement[0, 2:], 0.2)
        np.testing.assert_allclose(report.displacement[1], 0.3)

    def test_intermittent_new_sheet_preserves_detached_old_sheet(self):
        state = HiratoSheetShadow(ns=2)
        state.shed(
            step=1,
            leading_edges=self.le,
            gamma_now=np.array([0.2, 0.0]),
            active=np.array([True, False]),
            first_aft_edges=self.first,
        )
        old = state.rings.copy()
        old_sheet_id = int(state.sheet_id[0])
        second_first = self.first.copy()
        second_first[:, :, 0] = 0.15
        state.shed(
            step=5,
            leading_edges=self.le,
            gamma_now=np.array([0.4, 0.0]),
            active=np.array([True, False]),
            first_aft_edges=second_first,
            new_sheet=np.array([True, False]),
        )
        np.testing.assert_allclose(state.rings[0], old[0])
        self.assertNotEqual(int(state.sheet_id[1]), old_sheet_id)
        np.testing.assert_allclose(state.rings[1, 2:, 0], 0.15)

    def test_new_sheet_requires_an_active_strip(self):
        state = HiratoSheetShadow(ns=2)
        with self.assertRaises(HiratoEquationError):
            state.shed(
                step=0,
                leading_edges=self.le,
                gamma_now=np.zeros(2),
                active=np.array([False, False]),
                first_aft_edges=self.first,
                new_sheet=np.array([True, False]),
            )

    def test_moments_are_observations_of_spatial_state(self):
        state = HiratoSheetShadow(ns=2)
        state.shed(
            step=0,
            leading_edges=self.le,
            gamma_now=np.array([0.2, -0.1]),
            active=np.array([True, True]),
            first_aft_edges=self.first,
        )
        obs = state.strip_observables()
        np.testing.assert_allclose(obs["circulation"], [0.2, -0.1])
        np.testing.assert_array_equal(obs["ring_count"], [1, 1])
        self.assertTrue(np.all(np.isfinite(obs["centroid"])))

    def test_full_local_velocity_closes_channel_ledger(self):
        state = HiratoSheetShadow(ns=2)
        state.shed(
            step=0,
            leading_edges=self.le,
            gamma_now=np.array([0.2, -0.1]),
            active=np.array([True, True]),
            first_aft_edges=self.first,
        )
        bound = state.rings.copy()
        bound[..., 2] -= 0.2
        tev = state.rings[:1].copy()
        tev[..., 0] += 0.8
        ledger = state.full_local_velocity(
            bound_rings=bound,
            bound_gamma=np.array([0.05, -0.03]),
            tev_rings=tev,
            tev_gamma=np.array([0.02]),
            u_infinity=np.array([2.0, 0.0, 0.0]),
            core_radius=0.05,
            mirror_symmetry=False,
        )
        np.testing.assert_allclose(
            ledger.total,
            ledger.freestream + ledger.bound + ledger.tev + ledger.lev,
            rtol=0.0,
            atol=1e-14,
        )
        self.assertTrue(np.all(np.isfinite(ledger.total)))

    def test_full_local_convection_includes_mirrored_free_fields(self):
        state = HiratoSheetShadow(ns=2)
        state.shed(
            step=0,
            leading_edges=self.le,
            gamma_now=np.array([0.2, -0.1]),
            active=np.array([True, True]),
            first_aft_edges=self.first,
        )
        empty_rings = np.empty((0, 4, 3))
        empty_gamma = np.empty(0)
        no_mirror = state.full_local_velocity(
            bound_rings=empty_rings,
            bound_gamma=empty_gamma,
            tev_rings=empty_rings,
            tev_gamma=empty_gamma,
            u_infinity=np.array([1.0, 0.0, 0.0]),
            core_radius=0.05,
            mirror_symmetry=False,
        )
        with_mirror = state.full_local_velocity(
            bound_rings=empty_rings,
            bound_gamma=empty_gamma,
            tev_rings=empty_rings,
            tev_gamma=empty_gamma,
            u_infinity=np.array([1.0, 0.0, 0.0]),
            core_radius=0.05,
            mirror_symmetry=True,
        )
        self.assertGreater(
            float(np.max(np.abs(with_mirror.lev - no_mirror.lev))),
            1e-8,
        )
        before = state.rings.copy()
        ledger = state.convect_full_local(
            dt=0.01,
            bound_rings=empty_rings,
            bound_gamma=empty_gamma,
            tev_rings=empty_rings,
            tev_gamma=empty_gamma,
            u_infinity=np.array([1.0, 0.0, 0.0]),
            core_radius=0.05,
            mirror_symmetry=True,
        )
        np.testing.assert_allclose(state.rings, before + 0.01 * ledger.total)

    def test_full_local_eq24_reports_bootstrap_and_closes_displacement(self):
        state = HiratoSheetShadow(ns=2)
        state.shed(
            step=0,
            leading_edges=self.le,
            gamma_now=np.array([0.2, -0.1]),
            active=np.array([True, True]),
            first_aft_edges=self.first,
        )
        empty_rings = np.empty((0, 4, 3))
        empty_gamma = np.empty(0)
        before = state.rings.copy()
        ledger, report = state.convect_full_local_eq24(
            step=0,
            dt=0.01,
            bound_rings=empty_rings,
            bound_gamma=empty_gamma,
            tev_rings=empty_rings,
            tev_gamma=empty_gamma,
            u_infinity=np.array([1.0, 0.0, 0.0]),
            core_radius=0.05,
            mirror_symmetry=False,
        )
        np.testing.assert_allclose(state.rings, before + 0.01 * ledger.total)
        np.testing.assert_allclose(report.displacement, 0.01 * ledger.total)
        self.assertTrue(np.all(report.bootstrap_vertex))


if __name__ == "__main__":
    unittest.main()
