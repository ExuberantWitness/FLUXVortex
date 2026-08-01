import sys
import unittest
from pathlib import Path

import numpy as np

PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from claim_runtime.viscous_shell_geometry import (  # noqa: E402
    ViscousShellGeometryError,
    naca4_dual_surface_shell,
    rigidly_transform_shell,
)


class ViscousShellGeometryTests(unittest.TestCase):
    def setUp(self):
        beta = np.linspace(0.0, np.pi, 81)
        self.xi = 0.5*(1.0-np.cos(beta))
        self.span = np.linspace(0.0, 0.8, 9)
        self.chord = np.linspace(0.287, 0.12, len(self.span))
        self.shell = naca4_dual_surface_shell(
            self.xi, self.span, self.chord
        )

    def test_dual_wall_mean_is_exactly_n1_camber_surface(self):
        reconstructed = 0.5*(
            self.shell.upper_surface+self.shell.lower_surface
        )
        np.testing.assert_allclose(
            reconstructed, self.shell.mean_surface, atol=3.0e-17
        )
        norms = np.linalg.norm(
            self.shell.section_director, axis=2
        )
        np.testing.assert_allclose(norms, 1.0, atol=2.0e-16)

    def test_naca2406_maximum_total_thickness_is_six_percent(self):
        total_ratio = (
            2.0*self.shell.half_thickness[:, 0]/self.chord[0]
        )
        self.assertLess(abs(float(np.max(total_ratio))-0.06), 1.0e-4)
        location = self.xi[int(np.argmax(total_ratio))]
        self.assertLess(abs(float(location)-0.30), 0.02)
        self.assertEqual(self.shell.half_thickness[0, 0], 0.0)

    def test_rigid_transform_preserves_material_pairing_and_thickness(self):
        angle = 0.73
        rotation = np.array(
            [
                [np.cos(angle), 0.0, np.sin(angle)],
                [0.0, 1.0, 0.0],
                [-np.sin(angle), 0.0, np.cos(angle)],
            ]
        )
        moved = rigidly_transform_shell(
            self.shell,
            rotation=rotation,
            translation=[0.3, -0.2, 0.1],
        )
        np.testing.assert_allclose(
            0.5*(moved.upper_surface+moved.lower_surface),
            moved.mean_surface,
            atol=2.0e-16,
        )
        np.testing.assert_allclose(
            np.linalg.norm(
                moved.upper_surface-moved.lower_surface, axis=2
            ),
            2.0*self.shell.half_thickness,
            atol=2.0e-16,
        )

    def test_invalid_material_coordinates_and_rotation_fail(self):
        with self.assertRaises(ViscousShellGeometryError):
            naca4_dual_surface_shell(
                [0.0, 0.5, 0.4, 1.0], self.span, self.chord
            )
        with self.assertRaises(ViscousShellGeometryError):
            rigidly_transform_shell(
                self.shell,
                rotation=2.0*np.eye(3),
                translation=np.zeros(3),
            )


if __name__ == "__main__":
    unittest.main()
