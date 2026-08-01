import sys
import unittest
from pathlib import Path

import numpy as np

PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from unified_pressure_observation_guard import (  # noqa: E402
    PressureObservationError,
    integrate_profile_pressure,
)


class UnifiedPressureObservationTests(unittest.TestCase):
    def test_uniform_pressure_on_closed_clockwise_profile_is_zero(self):
        coordinates = np.array([
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 0.0],
            [0.0, 0.0],
        ])
        pressure = np.full((3, 5), 2.7)
        cn, ct, cm = integrate_profile_pressure(coordinates, pressure)
        np.testing.assert_allclose(cn, 0.0, atol=2e-15)
        np.testing.assert_allclose(ct, 0.0, atol=2e-15)
        np.testing.assert_allclose(cm, 0.0, atol=2e-15)

    def test_excluded_segment_is_explicit_not_a_pressure_correction(self):
        coordinates = np.array([
            [0.0, 0.0],
            [1.0, 0.2],
            [1.0, -0.2],
            [0.0, 0.0],
        ])
        pressure = np.array([0.0, -1.0, -3.0, 0.0])
        _, ct_closed, _ = integrate_profile_pressure(
            coordinates, pressure
        )
        _, ct_profile, _ = integrate_profile_pressure(
            coordinates,
            pressure,
            excluded_segments=(1,),
        )
        expected_base_contribution = (
            0.5 * (pressure[1] + pressure[2])
            * (coordinates[2, 1] - coordinates[1, 1])
        )
        np.testing.assert_allclose(
            ct_closed - ct_profile,
            expected_base_contribution,
            atol=2e-15,
        )

    def test_shape_and_nonfinite_values_fail(self):
        coordinates = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 0.0],
        ])
        with self.assertRaises(PressureObservationError):
            integrate_profile_pressure(coordinates, np.ones(2))
        with self.assertRaises(PressureObservationError):
            integrate_profile_pressure(
                coordinates,
                np.array([0.0, np.nan, 0.0]),
            )
        with self.assertRaises(PressureObservationError):
            integrate_profile_pressure(
                coordinates,
                np.ones(3),
                excluded_segments=(7,),
            )


if __name__ == "__main__":
    unittest.main()
