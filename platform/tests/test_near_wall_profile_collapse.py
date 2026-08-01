import sys
import unittest
from pathlib import Path

import numpy as np

PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from claim_runtime.near_wall_profile_collapse import (  # noqa: E402
    NearWallProfile,
    NearWallProfileError,
    project_near_wall_profile,
)


def profile(
    coordinate,
    velocity_x,
    *,
    density=None,
    normal=(0.0, 0.0, 1.0),
    external=(1.0, 0.0, 0.0),
    plus=(1.0, 0.0, -0.08),
    minus=(0.0, 0.0, 0.0),
    edge="manufactured-explicit-edge",
):
    coordinate = np.asarray(coordinate, dtype=float)
    velocity_x = np.asarray(velocity_x, dtype=float)
    velocity = np.zeros((len(coordinate), 3))
    velocity[:, 0] = velocity_x
    if density is None:
        density = np.ones(len(coordinate))
    return NearWallProfile(
        normal_coordinate=coordinate,
        density=density,
        velocity=velocity,
        external_tangential_velocity=external,
        outer_velocity_plus=plus,
        outer_velocity_minus=minus,
        sheet_normal=normal,
        edge_convention=edge,
    )


class NearWallProfileCollapseTests(unittest.TestCase):
    def test_analytic_linear_profile(self):
        coordinate = np.linspace(0.0, 1.0, 5001)
        result = project_near_wall_profile(profile(coordinate, coordinate))
        self.assertAlmostEqual(result.mass_flux_defect[0], 0.5, delta=2.0e-7)
        self.assertAlmostEqual(
            result.momentum_flux_trace,
            1.0/6.0,
            delta=2.0e-7,
        )
        self.assertAlmostEqual(result.surface_mass_density, 1.0, delta=2.0e-7)
        self.assertAlmostEqual(result.surface_momentum[0], 0.5, delta=2.0e-7)
        np.testing.assert_allclose(
            result.vortex_sheet_strength,
            [0.0, 1.0, 0.0],
            atol=2.0e-15,
        )
        self.assertAlmostEqual(result.entrainment_strength, 0.08)

    def test_rotation_covariance(self):
        coordinate = np.linspace(0.0, 0.7, 3001)
        velocity = np.zeros((len(coordinate), 3))
        velocity[:, 0] = 1.3*(coordinate/coordinate[-1])**0.7
        velocity[:, 1] = -0.2*(coordinate/coordinate[-1])
        original = NearWallProfile(
            normal_coordinate=coordinate,
            density=1.1+0.05*coordinate,
            velocity=velocity,
            external_tangential_velocity=[1.3, -0.2, 0.0],
            outer_velocity_plus=[1.3, -0.2, -0.07],
            outer_velocity_minus=[0.1, 0.05, 0.02],
            sheet_normal=[0.0, 0.0, 1.0],
            edge_convention="manufactured-rotating-edge",
        )
        reference = project_near_wall_profile(original)

        axis = np.array([0.3, -0.8, 0.4])
        axis /= np.linalg.norm(axis)
        angle = 0.73
        cross = np.array([
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ])
        rotation = (
            np.eye(3)*np.cos(angle)
            +(1.0-np.cos(angle))*np.outer(axis, axis)
            +np.sin(angle)*cross
        )
        moved = NearWallProfile(
            normal_coordinate=coordinate,
            density=original.density,
            velocity=original.velocity@rotation.T,
            external_tangential_velocity=
                original.external_tangential_velocity@rotation.T,
            outer_velocity_plus=original.outer_velocity_plus@rotation.T,
            outer_velocity_minus=original.outer_velocity_minus@rotation.T,
            sheet_normal=original.sheet_normal@rotation.T,
            edge_convention=original.edge_convention,
        )
        transformed = project_near_wall_profile(moved)
        np.testing.assert_allclose(
            transformed.mass_flux_defect,
            reference.mass_flux_defect@rotation.T,
            atol=2.0e-12,
        )
        np.testing.assert_allclose(
            transformed.momentum_flux_defect,
            rotation@reference.momentum_flux_defect@rotation.T,
            atol=2.0e-12,
        )
        np.testing.assert_allclose(
            transformed.surface_momentum,
            reference.surface_momentum@rotation.T,
            atol=2.0e-12,
        )
        np.testing.assert_allclose(
            transformed.vortex_sheet_strength,
            reference.vortex_sheet_strength@rotation.T,
            atol=2.0e-12,
        )
        self.assertAlmostEqual(
            transformed.entrainment_strength,
            reference.entrainment_strength,
            delta=2.0e-12,
        )

    def test_outer_plateau_is_invisible_to_deficits_but_not_ves(self):
        base_coordinate = np.linspace(0.0, 1.0, 5001)
        base = project_near_wall_profile(
            profile(base_coordinate, base_coordinate, edge="edge-at-one")
        )
        extension = np.linspace(1.0002, 1.2, 1000)
        extended_coordinate = np.concatenate((base_coordinate, extension))
        extended_velocity = np.concatenate((
            base_coordinate,
            np.ones(len(extension)),
        ))
        extended = project_near_wall_profile(profile(
            extended_coordinate,
            extended_velocity,
            edge="edge-including-uniform-outer-plateau",
        ))
        np.testing.assert_allclose(
            extended.mass_flux_defect,
            base.mass_flux_defect,
            atol=2.0e-7,
        )
        np.testing.assert_allclose(
            extended.momentum_flux_defect,
            base.momentum_flux_defect,
            atol=2.0e-7,
        )
        self.assertGreater(
            extended.surface_mass_density-base.surface_mass_density,
            0.19,
        )
        self.assertGreater(
            extended.surface_momentum[0]-base.surface_momentum[0],
            0.19,
        )
        self.assertNotEqual(extended.edge_convention, base.edge_convention)

    def test_equal_deficit_profiles_have_distinct_ves_states(self):
        coordinate_a = np.linspace(0.0, 1.0, 20001)
        result_a = project_near_wall_profile(
            profile(coordinate_a, coordinate_a, edge="profile-a")
        )

        delta_b = 15.0/14.0
        coordinate_b = np.linspace(0.0, delta_b, 24001)
        eta = coordinate_b/delta_b
        velocity_b = (
            eta
            +0.2*eta*(1.0-eta)
            -0.2819747927945442*eta*(1.0-eta)*(1.0-2.0*eta)
        )
        self.assertGreaterEqual(np.min(np.diff(velocity_b)), -1.0e-14)
        result_b = project_near_wall_profile(
            profile(coordinate_b, velocity_b, edge="profile-b")
        )

        np.testing.assert_allclose(
            result_b.mass_flux_defect,
            result_a.mass_flux_defect,
            atol=2.0e-7,
        )
        self.assertAlmostEqual(
            result_b.momentum_flux_trace,
            result_a.momentum_flux_trace,
            delta=2.0e-7,
        )
        self.assertGreater(
            result_b.surface_mass_density-result_a.surface_mass_density,
            0.07,
        )
        self.assertGreater(
            result_b.surface_momentum[0]-result_a.surface_momentum[0],
            0.07,
        )

    def test_invalid_inputs_fail_explicitly(self):
        valid = dict(
            normal_coordinate=[0.0, 1.0],
            density=[1.0, 1.0],
            velocity=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            external_tangential_velocity=[1.0, 0.0, 0.0],
            outer_velocity_plus=[1.0, 0.0, 0.0],
            outer_velocity_minus=[0.0, 0.0, 0.0],
            sheet_normal=[0.0, 0.0, 1.0],
            edge_convention="explicit",
        )
        for replacement in (
            {"normal_coordinate": [0.0, 0.0]},
            {"density": [1.0, 0.0]},
            {"sheet_normal": [0.0, 0.0, 2.0]},
            {"external_tangential_velocity": [1.0, 0.0, 0.1]},
            {"edge_convention": ""},
        ):
            bad = dict(valid)
            bad.update(replacement)
            with self.assertRaises(NearWallProfileError):
                NearWallProfile(**bad)


if __name__ == "__main__":
    unittest.main()
