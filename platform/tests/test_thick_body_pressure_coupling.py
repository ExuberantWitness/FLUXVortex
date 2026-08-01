import sys
import unittest
from pathlib import Path

import numpy as np

PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from thick_body_pressure_coupling_guard import (  # noqa: E402
    ThickBodyPressureCouplingError,
    circular_cylinder_pressure_coupling,
)


class ThickBodyPressureCouplingTests(unittest.TestCase):
    def evaluate(self, ratio=0.35, gauge=0.731):
        return circular_cylinder_pressure_coupling(
            radius=1.0,
            freestream_speed=1.0,
            circulation_velocity_ratio=ratio,
            azimuth_nodes=1024,
            pressure_gauge=gauge,
        )

    def test_total_pressure_contains_exact_bernoulli_cross_term(self):
        result = self.evaluate()
        np.testing.assert_allclose(
            result.total_pressure_coefficient
            - result.pressure_level_addition,
            result.bernoulli_cross_term,
            rtol=0.0,
            atol=2.0e-15,
        )
        self.assertAlmostEqual(
            float(
                np.max(
                    np.abs(
                        result.total_pressure_coefficient
                        - result.pressure_level_addition
                    )
                )
            ),
            4.0 * 0.35,
            places=13,
        )

    def test_pressure_level_addition_loses_kutta_joukowski_lift(self):
        result = self.evaluate()
        self.assertAlmostEqual(
            float(result.total_force_coefficient[0]), 0.0, places=13
        )
        self.assertAlmostEqual(
            float(result.total_force_coefficient[1]),
            -2.0 * np.pi * 0.35,
            places=13,
        )
        np.testing.assert_allclose(
            result.pressure_addition_force_coefficient,
            0.0,
            rtol=0.0,
            atol=2.0e-14,
        )

    def test_uniform_pressure_gauge_is_force_free(self):
        reference = self.evaluate(gauge=0.0)
        shifted = self.evaluate(gauge=91.7)
        np.testing.assert_allclose(
            shifted.total_force_coefficient,
            reference.total_force_coefficient,
            rtol=0.0,
            atol=2.0e-13,
        )

    def test_invalid_canonical_inputs_fail(self):
        with self.assertRaises(ThickBodyPressureCouplingError):
            circular_cylinder_pressure_coupling(
                radius=-1.0,
                freestream_speed=1.0,
                circulation_velocity_ratio=0.2,
                azimuth_nodes=1024,
            )
        with self.assertRaises(ThickBodyPressureCouplingError):
            circular_cylinder_pressure_coupling(
                radius=1.0,
                freestream_speed=1.0,
                circulation_velocity_ratio=0.2,
                azimuth_nodes=1025,
            )


if __name__ == "__main__":
    unittest.main()
