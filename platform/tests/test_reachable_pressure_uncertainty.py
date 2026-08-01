"""Unit tests for S3ai-v2 same-space uncertainty; no physics run occurs."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from claim_runtime.reachable_pressure_uncertainty import (  # noqa: E402
    ReachablePressureUncertaintyError,
    adjacent_mixed_cube,
    contracting_quadrature_tail,
    resolved_norm_interval,
    second_order_richardson,
)


class ReachablePressureUncertaintyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mass = np.array(
            ((2.0, 0.25), (0.25, 1.5)),
            dtype=float,
        )
        self.limit = np.array((1.0, -0.5))
        self.error = np.array((0.2, 0.4))

    def test_second_order_family_contracts_and_exact_zero_plateaus(self) -> None:
        report = second_order_richardson(
            fine=self.limit + self.error,
            medium=self.limit + 4.0 * self.error,
            coarse=self.limit + 16.0 * self.error,
            mass=self.mass,
        )
        self.assertTrue(report.passed)
        self.assertFalse(report.plateau)
        self.assertAlmostEqual(report.contraction_ratio, 4.0)
        np.testing.assert_allclose(
            report.fine_extrapolated,
            self.limit,
            rtol=0.0,
            atol=4.0e-16,
        )
        expected_allowance = (
            np.sqrt(
                (report.fine_extrapolated - (self.limit + self.error))
                @ np.linalg.solve(
                    self.mass,
                    report.fine_extrapolated
                    - (self.limit + self.error),
                )
            )
            + np.sqrt(
                (report.fine_extrapolated - report.medium_extrapolated)
                @ np.linalg.solve(
                    self.mass,
                    report.fine_extrapolated
                    - report.medium_extrapolated,
                )
            )
        )
        self.assertAlmostEqual(report.allowance, expected_allowance)
        zero = second_order_richardson(
            fine=np.zeros(2),
            medium=np.zeros(2),
            coarse=np.zeros(2),
            mass=self.mass,
        )
        self.assertTrue(zero.passed)
        self.assertTrue(zero.plateau)
        self.assertEqual(zero.allowance, 0.0)

    def test_quadrature_tail_requires_contraction(self) -> None:
        converged = contracting_quadrature_tail(
            coarse=self.limit + 4.0 * self.error,
            medium=self.limit + 2.0 * self.error,
            fine=self.limit + self.error,
            mass=self.mass,
        )
        self.assertTrue(converged.passed)
        self.assertAlmostEqual(converged.contraction_ratio, 0.5)
        self.assertAlmostEqual(
            converged.allowance,
            converged.medium_fine_change
            / (1.0 - converged.contraction_ratio),
        )
        failed = contracting_quadrature_tail(
            coarse=self.limit + self.error,
            medium=self.limit + 2.0 * self.error,
            fine=self.limit + 4.0 * self.error,
            mass=self.mass,
        )
        self.assertFalse(failed.passed)
        self.assertTrue(np.isinf(failed.allowance))

    def test_complete_cube_recovers_pairwise_and_three_way_terms(self) -> None:
        base = self.limit
        e = np.array((0.1, 0.0))
        t = np.array((0.0, 0.2))
        q = np.array((0.05, -0.05))
        et = np.array((0.03, 0.04))
        eq = np.array((-0.02, 0.01))
        tq = np.array((0.01, -0.03))
        etq = np.array((0.015, 0.025))
        values = {}
        for index in range(8):
            key = f"{index:03b}"
            i, j, k = (int(value) for value in key)
            values[key] = (
                base
                + i * e
                + j * t
                + k * q
                + i * j * et
                + i * k * eq
                + j * k * tq
                + i * j * k * etq
            )
        report = adjacent_mixed_cube(values, self.mass)
        np.testing.assert_allclose(report.epsilon_timestep, et)
        np.testing.assert_allclose(report.epsilon_quadrature, eq)
        np.testing.assert_allclose(report.timestep_quadrature, tq)
        np.testing.assert_allclose(
            report.epsilon_timestep_quadrature,
            etq,
        )
        self.assertAlmostEqual(
            report.allowance,
            sum(report.component_norms.values()),
        )

    def test_resolved_interval_never_has_a_negative_lower_bound(self) -> None:
        interval = resolved_norm_interval(
            np.array((0.1, -0.2)),
            self.mass,
            uncertainty=10.0,
        )
        self.assertEqual(interval.lower, 0.0)
        self.assertGreater(interval.upper, interval.observed_norm)

    def test_empty_or_nonsymmetric_spaces_fail_closed(self) -> None:
        with self.assertRaises(ReachablePressureUncertaintyError):
            resolved_norm_interval(
                np.empty(0),
                np.empty((0, 0)),
                uncertainty=0.0,
            )
        with self.assertRaises(ReachablePressureUncertaintyError):
            resolved_norm_interval(
                np.ones(2),
                np.array(((1.0, 1.0e-15), (0.0, 1.0))),
                uncertainty=0.0,
            )


if __name__ == "__main__":
    unittest.main()
