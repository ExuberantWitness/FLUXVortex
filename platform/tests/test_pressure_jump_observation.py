import sys
import unittest
from pathlib import Path

import numpy as np

PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from pressure_jump_observation_guard import (  # noqa: E402
    PressureJumpObservationError,
    integrate_normal_pressure_jump,
    pair_pressure_jump,
    paired_common_mode_profile,
)
from unified_pressure_observation_guard import (  # noqa: E402
    integrate_profile_pressure,
)


class PressureJumpObservationTests(unittest.TestCase):
    def setUp(self):
        self.coordinates = np.array([
            [0.0, 0.0],
            [0.5, 0.2],
            [1.0, 0.1],
            [1.0, -0.1],
            [0.5, -0.15],
            [0.0, 0.0],
        ])
        self.profile = np.array([
            [-1.0, -2.0, -0.4, 0.2, 0.1, 0.0],
            [-0.8, -1.5, -0.3, 0.3, 0.2, 0.1],
        ])

    def test_paired_jump_reconstructs_normal_profile_force(self):
        x, _, _, jump = pair_pressure_jump(
            self.coordinates,
            self.profile,
            side_point_count=3,
        )
        cn_jump = integrate_normal_pressure_jump(x, jump)
        cn_profile, _, _ = integrate_profile_pressure(
            self.coordinates,
            self.profile,
            excluded_segments=(2,),
        )
        np.testing.assert_allclose(cn_jump, cn_profile, atol=2e-15)

    def test_common_side_mode_preserves_jump_but_changes_thick_Ct(self):
        x, _, _, jump = pair_pressure_jump(
            self.coordinates,
            self.profile,
            side_point_count=3,
        )
        common = np.array([
            [0.0, 0.2, 0.5],
            [0.0, -0.1, 0.3],
        ])
        shifted = self.profile + paired_common_mode_profile(common)
        _, _, _, shifted_jump = pair_pressure_jump(
            self.coordinates,
            shifted,
            side_point_count=3,
        )
        np.testing.assert_allclose(shifted_jump, jump, atol=2e-15)
        _, ct_original, _ = integrate_profile_pressure(
            self.coordinates,
            self.profile,
            excluded_segments=(2,),
        )
        _, ct_shifted, _ = integrate_profile_pressure(
            self.coordinates,
            shifted,
            excluded_segments=(2,),
        )
        self.assertGreater(np.max(np.abs(ct_shifted - ct_original)), 0.01)
        np.testing.assert_allclose(
            integrate_normal_pressure_jump(x, shifted_jump),
            integrate_normal_pressure_jump(x, jump),
            atol=2e-15,
        )

    def test_unpaired_or_nonmonotone_coordinates_fail(self):
        bad = self.coordinates.copy()
        bad[4, 0] = 0.6
        with self.assertRaises(PressureJumpObservationError):
            pair_pressure_jump(
                bad,
                self.profile,
                side_point_count=3,
            )
        with self.assertRaises(PressureJumpObservationError):
            integrate_normal_pressure_jump(
                np.array([0.0, 0.5, 0.4]),
                np.ones((1, 3)),
            )


if __name__ == "__main__":
    unittest.main()
