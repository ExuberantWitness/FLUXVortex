"""Manufactured tests for frozen S3ai-v2.2 parity primitives."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from claim_runtime.reachable_pressure_symmetry import (  # noqa: E402
    MANUFACTURED_EVEN_ACTIVE,
    MANUFACTURED_ODD_ACTIVE,
    MANUFACTURED_ODD_AMPLITUDE,
    ReachablePressureSymmetryError,
    canonical_active_reflection,
    parity_decomposition,
    parity_quadrature_decision,
    project_active_parity,
    projected_quadrature_interval,
    span_parity_operators,
    stagewise_odd_noncancellation,
    typed_mass_norm,
    unprojected_operand_round_term,
)


def _persymmetric_spd_mass() -> np.ndarray:
    mass = np.diag((2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0))
    off_diagonal = (0.125, 0.25, 0.375, 0.375, 0.25, 0.125)
    for index, value in enumerate(off_diagonal):
        mass[index, index + 1] = value
        mass[index + 1, index] = value
    return mass


class ReachablePressureSymmetryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mass = _persymmetric_spd_mass()
        self.even = MANUFACTURED_EVEN_ACTIVE.copy()
        self.odd = MANUFACTURED_ODD_ACTIVE.copy()

    def test_canonical_reversal_projectors_and_mass_contract(self) -> None:
        operators = span_parity_operators(self.mass)
        np.testing.assert_array_equal(
            operators.reflection @ self.even,
            self.even,
        )
        np.testing.assert_array_equal(
            operators.reflection @ self.odd,
            -self.odd,
        )
        np.testing.assert_array_equal(
            operators.even_projector @ operators.odd_projector,
            np.zeros((7, 7)),
        )
        with self.assertRaises(ReachablePressureSymmetryError):
            span_parity_operators(
                self.mass,
                reflection=np.eye(7),
            )
        wrong = canonical_active_reflection()
        wrong[[0, 1]] = wrong[[1, 0]]
        with self.assertRaises(ReachablePressureSymmetryError):
            span_parity_operators(self.mass, reflection=wrong)

    def test_mass_must_be_exact_symmetric_spd_and_persymmetric(self) -> None:
        nonsymmetric = self.mass.copy()
        nonsymmetric[0, 1] += np.finfo(float).eps
        with self.assertRaises(ReachablePressureSymmetryError):
            span_parity_operators(nonsymmetric)
        nonpersymmetric = self.mass.copy()
        nonpersymmetric[0, 0] += 0.25
        with self.assertRaises(ReachablePressureSymmetryError):
            span_parity_operators(nonpersymmetric)
        indefinite = np.eye(7)
        indefinite[3, 3] = -1.0
        with self.assertRaises(ReachablePressureSymmetryError):
            span_parity_operators(indefinite)

    def test_manufactured_primal_and_dual_components_are_recovered(self) -> None:
        amplitude = MANUFACTURED_ODD_AMPLITUDE
        primal = self.even + amplitude * self.odd
        np.testing.assert_array_equal(
            project_active_parity(
                primal,
                self.mass,
                parity="even",
            ),
            self.even,
        )
        np.testing.assert_array_equal(
            project_active_parity(
                primal,
                self.mass,
                parity="odd",
            ),
            amplitude * self.odd,
        )

        dual = self.mass @ self.even + amplitude * self.mass @ self.odd
        np.testing.assert_allclose(
            project_active_parity(
                dual,
                self.mass,
                parity="even",
            ),
            self.mass @ self.even,
            rtol=0.0,
            atol=2.0e-15,
        )
        np.testing.assert_allclose(
            project_active_parity(
                dual,
                self.mass,
                parity="odd",
            ),
            amplitude * self.mass @ self.odd,
            rtol=0.0,
            atol=2.0e-15,
        )

    def test_primal_and_dual_pythagorean_identities(self) -> None:
        primal = parity_decomposition(
            self.even + 0.25 * self.odd,
            self.mass,
            metric_role="primal",
        )
        dual = parity_decomposition(
            self.mass @ self.even + 0.25 * self.mass @ self.odd,
            self.mass,
            metric_role="dual",
        )
        for report in (primal, dual):
            scale = max(report.value_norm * report.value_norm, 1.0)
            self.assertLessEqual(
                report.pythagorean_abs_residual,
                64.0 * np.finfo(float).eps * scale,
            )

    def test_projected_tail_adds_unprojected_round_term_exactly_once(self) -> None:
        error = 0.125 * self.odd
        q8 = self.even + 4.0 * error
        q10 = self.even + 2.0 * error
        q12 = self.even + error
        operands = (q8, q10, q12)
        report = projected_quadrature_interval(
            q8=q8,
            q10=q10,
            q12=q12,
            mass=self.mass,
            parity="odd",
            metric_role="primal",
            unprojected_round_operands=operands,
        )
        round_term = unprojected_operand_round_term(
            operands,
            self.mass,
            metric_role="primal",
        )
        self.assertTrue(report.passed)
        self.assertFalse(report.floating_plateau)
        self.assertAlmostEqual(report.contraction_ratio, 0.5)
        self.assertEqual(report.floating_round_term, round_term)
        self.assertEqual(
            report.uncertainty,
            report.tail_allowance + round_term,
        )
        self.assertNotEqual(
            report.uncertainty,
            report.tail_allowance + 2.0 * round_term,
        )

    def test_noncontracting_required_projection_is_protocol_no_go(self) -> None:
        q8 = self.even + 0.125 * self.odd
        q10 = self.even + 0.25 * self.odd
        q12 = self.even + 0.5 * self.odd
        report = parity_quadrature_decision(
            q8=q8,
            q10=q10,
            q12=q12,
            mass=self.mass,
            metric_role="primal",
            unprojected_round_operands=(q8, q10, q12),
        )
        self.assertTrue(report.protocol_no_go)
        self.assertFalse(report.odd.passed)
        self.assertIsNone(report.odd.lower)
        self.assertIn(
            "odd projected q family did not contract or plateau",
            report.reasons,
        )

    def test_zero_and_tiny_even_have_no_resolved_symmetry_violation(self) -> None:
        zero = np.zeros(7)
        zero_report = parity_quadrature_decision(
            q8=zero,
            q10=zero,
            q12=zero,
            mass=self.mass,
            metric_role="primal",
            unprojected_round_operands=(zero, zero, zero),
        )
        self.assertFalse(zero_report.protocol_no_go)
        self.assertTrue(zero_report.no_resolved_symmetry_violation)
        self.assertEqual(zero_report.odd.lower, 0.0)
        self.assertIsNone(zero_report.relative_odd_to_even_upper)

        tiny_even = (2.0 ** -40) * self.even
        tiny_report = parity_quadrature_decision(
            q8=tiny_even,
            q10=tiny_even,
            q12=tiny_even,
            mass=self.mass,
            metric_role="primal",
            unprojected_round_operands=(
                tiny_even,
                tiny_even,
                tiny_even,
            ),
        )
        self.assertFalse(tiny_report.protocol_no_go)
        self.assertTrue(tiny_report.no_resolved_symmetry_violation)
        self.assertEqual(tiny_report.odd.lower, 0.0)
        self.assertGreater(tiny_report.even.lower, 0.0)
        self.assertIsNotNone(tiny_report.relative_odd_to_even_upper)

    def test_tiny_odd_fails_only_through_positive_odd_lower_bound(self) -> None:
        tiny_odd = (2.0 ** -40) * self.odd
        report = parity_quadrature_decision(
            q8=tiny_odd,
            q10=tiny_odd,
            q12=tiny_odd,
            mass=self.mass,
            metric_role="primal",
            unprojected_round_operands=(tiny_odd, tiny_odd, tiny_odd),
        )
        self.assertTrue(report.odd.passed)
        self.assertGreater(report.odd.lower, 0.0)
        self.assertTrue(report.protocol_no_go)
        self.assertTrue(report.resolved_symmetry_failure)
        self.assertFalse(report.no_resolved_symmetry_violation)
        self.assertIsNone(report.relative_odd_to_even_upper)
        self.assertEqual(
            report.reasons,
            ("odd lower bound L_minus is positive",),
        )

    def test_relative_bound_exists_only_for_resolved_even_signal(self) -> None:
        value = self.even + MANUFACTURED_ODD_AMPLITUDE * self.odd
        report = parity_quadrature_decision(
            q8=value,
            q10=value,
            q12=value,
            mass=self.mass,
            metric_role="primal",
            unprojected_round_operands=(value, value, value),
        )
        self.assertGreater(report.even.lower, 0.0)
        self.assertIsNotNone(report.relative_odd_to_even_upper)
        self.assertAlmostEqual(
            report.relative_odd_to_even_upper,
            report.odd.upper / report.even.lower,
        )

    def test_stagewise_odd_defects_cannot_cancel_in_window_sum(self) -> None:
        residuals = np.stack((self.odd, -self.odd))
        report = stagewise_odd_noncancellation(
            residuals,
            self.mass,
        )
        expected = 2.0 * typed_mass_norm(
            self.odd,
            self.mass,
            metric_role="dual",
        )
        self.assertAlmostEqual(report.value, expected)
        self.assertGreater(report.value, 0.0)
        self.assertEqual(report.window_odd_norm, 0.0)
        np.testing.assert_array_equal(report.window_residual, np.zeros(7))

        even_report = stagewise_odd_noncancellation(
            np.stack((self.even, -self.even)),
            self.mass,
        )
        self.assertEqual(even_report.value, 0.0)


if __name__ == "__main__":
    unittest.main()
