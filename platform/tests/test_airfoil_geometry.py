import sys
import unittest
from pathlib import Path

import numpy as np

PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

import airfoil_geometry as ag  # noqa: E402


class AirfoilGeometryTests(unittest.TestCase):
    def test_sd7003_chord_frame_has_exact_geometric_endpoints(self):
        coordinates = ag.sd7003_chord_coordinates()
        i_le = int(np.argmin(coordinates[:, 0]))
        np.testing.assert_allclose(coordinates[i_le], (0.0, 0.0), atol=1e-14)
        self.assertAlmostEqual(float(coordinates[0, 0]), 1.0)
        self.assertAlmostEqual(float(coordinates[-1, 0]), 1.0)
        self.assertLess(abs(float(np.mean(coordinates[[0, -1], 1]))), 1e-14)

    def test_sd7003_mean_camber_is_from_two_surfaces(self):
        x = np.linspace(0.0, 1.0, 401)
        camber = ag.sd7003_mean_camber(x)
        self.assertTrue(np.all(np.isfinite(camber)))
        self.assertGreater(float(np.max(camber)), 0.01)
        self.assertLess(float(np.max(camber)), 0.02)
        self.assertLess(abs(float(camber[0])), 1e-14)
        self.assertLess(abs(float(camber[-1])), 1e-4)

    def test_sd7003_wing_has_requested_planform_and_camber(self):
        wing = ag.sd7003_mean_camber_wing(8, 16, chord=0.1, half_span=0.3)
        self.assertEqual(wing.shape, (9, 17, 3))
        np.testing.assert_allclose(wing[:, 0, 2], wing[:, -1, 2])
        self.assertAlmostEqual(float(wing[-1, -1, 0]), 0.1)
        self.assertAlmostEqual(float(wing[0, -1, 1]), 0.3)


if __name__ == "__main__":
    unittest.main()
