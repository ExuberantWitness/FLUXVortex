"""Synthetic/API guards for the independent FluxV v5b Ptera history adapter."""

from __future__ import annotations

import unittest

import numpy as np

from forward_flight_benchmarks.fluxv_v5b_ptera_adapter import (
    build_crosspaper_smoke_input,
    coefficients_from_gp1_force,
)


class TestFluxVV5BPteraAdapter(unittest.TestCase):
    def test_three_declared_smokes_have_expected_full_structured_histories(self):
        expected = {
            "yang2025": ((40, 3, 5, 3), 20, np.arange(20, 40), 0.0325),
            "izraelevitz2017_fig14": (
                (193, 3, 13, 3),
                64,
                np.arange(128, 192),
                0.1016 * 0.3048,
            ),
            "baik2012": ((64, 3, 9, 3), 32, np.arange(32, 64), 0.076 * 0.600),
        }
        for case_id, (shape, steps, indices, area) in expected.items():
            with self.subTest(case_id=case_id):
                adapted = build_crosspaper_smoke_input(case_id)
                history = adapted.history
                self.assertEqual(history.corners_history.shape, shape)
                self.assertEqual(history.corner_velocity_history.shape, shape)
                self.assertEqual(history.steps_per_cycle, steps)
                np.testing.assert_array_equal(history.final_cycle_indices, indices)
                self.assertAlmostEqual(history.reference_area_m2, area, places=14)
                self.assertTrue(np.all(np.isfinite(history.corners_history)))
                self.assertTrue(np.all(np.isfinite(history.corner_velocity_history)))
                self.assertFalse(history.hirato_mirror_symmetry)

    def test_velocity_is_actual_material_backward_difference(self):
        adapted = build_crosspaper_smoke_input("yang2025")
        history = adapted.history
        np.testing.assert_array_equal(history.corner_velocity_history[0], 0.0)
        np.testing.assert_allclose(
            history.corner_velocity_history[1:],
            np.diff(history.corners_history, axis=0) / history.delta_time_s,
            rtol=0.0,
            atol=0.0,
        )

    def test_periodic_step_zero_velocity_is_explicit_opt_in(self):
        adapted = build_crosspaper_smoke_input(
            "baik2012", initial_velocity_mode="periodic_backward"
        )
        history = adapted.history
        expected = (
            history.corners_history[0]
            - history.corners_history[history.steps_per_cycle - 1]
        ) / history.delta_time_s
        np.testing.assert_allclose(
            history.corner_velocity_history[0], expected, rtol=0.0, atol=0.0
        )
        self.assertGreater(float(np.max(np.abs(expected))), 0.0)

    def test_ptera_type4_mesh_is_already_full_span(self):
        adapted = build_crosspaper_smoke_input("izraelevitz2017_fig14")
        history = adapted.history
        y = history.corners_history[0, :, :, 1]
        self.assertTrue(history.ptera_symmetric_full_mesh)
        self.assertLess(float(np.min(y)), 0.0)
        self.assertGreater(float(np.max(y)), 0.0)
        self.assertAlmostEqual(float(np.min(y)), -0.1524, places=12)
        self.assertAlmostEqual(float(np.max(y)), 0.1524, places=12)

    def test_force_signs_match_existing_benchmark_extractor(self):
        history = build_crosspaper_smoke_input("yang2025").history
        # GP1-to-wind flips x and z in the zero-alpha OperatingPoint.  Thus this GP1
        # force is positive thrust and positive lift in the benchmark convention.
        result = coefficients_from_gp1_force(np.array([-2.0, 0.0, 3.0]), history)
        self.assertAlmostEqual(float(result["thrust_n"]), 2.0)
        self.assertAlmostEqual(float(result["drag_n"]), -2.0)
        self.assertAlmostEqual(float(result["lift_n"]), 3.0)
        self.assertAlmostEqual(float(result["CD"]), -float(result["CT"]))

    def test_freestream_is_ptera_gp1_vector_not_wind_axis_scalar(self):
        for case_id in ("yang2025", "izraelevitz2017_fig14", "baik2012"):
            with self.subTest(case_id=case_id):
                history = build_crosspaper_smoke_input(case_id).history
                self.assertGreater(history.u_infinity_gp1_m_s[0], 0.0)
                self.assertAlmostEqual(history.u_infinity_gp1_m_s[1], 0.0, places=14)
                self.assertAlmostEqual(history.u_infinity_gp1_m_s[2], 0.0, places=12)


if __name__ == "__main__":
    unittest.main()
