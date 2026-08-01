import sys
import unittest
from pathlib import Path

import numpy as np

PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from claim_runtime.moving_separation_flux import (  # noqa: E402
    moving_separation_ibl_flux,
)
from claim_runtime.surface_ibl_state import (  # noqa: E402
    SurfaceIBLError,
    SurfaceIBLFields,
    surface_ibl_physical_flux,
)


def fields() -> SurfaceIBLFields:
    return SurfaceIBLFields(
        mass_flux_defect=[[0.4, -0.2, 0.0]],
        momentum_flux_defect=[[
            [0.6, 0.15, 0.0],
            [-0.1, 0.35, 0.0],
            [0.0, 0.0, 0.0],
        ]],
        kinetic_energy_defect_flux=[[0.8, -0.3, 0.0]],
        external_tangential_velocity=[[1.7, -0.4, 0.0]],
        external_velocity_surface_gradient=np.zeros((1, 3, 3)),
        wall_shear_over_density=np.zeros((1, 3)),
        dissipation_integral=np.zeros(1),
        surface_normal=[[0.0, 0.0, 1.0]],
    )


class MovingSeparationFluxTests(unittest.TestCase):
    def test_zero_relative_speed_reduces_to_physical_flux(self):
        state = fields()
        conormal = [[1.0, 0.0, 0.0]]
        physical = surface_ibl_physical_flux(
            state,
            outward_surface_conormal=conormal,
            edge_measure=1.3,
        )
        moving = moving_separation_ibl_flux(
            state,
            outward_surface_conormal=conormal,
            relative_conormal_speed=0.0,
            edge_measure=1.3,
        )
        np.testing.assert_allclose(
            moving.relative_momentum_defect_out,
            physical.momentum_out,
            atol=0.0,
        )
        np.testing.assert_allclose(
            moving.relative_energy_defect_out,
            physical.energy_out,
            atol=0.0,
        )

    def test_boundary_motion_is_signed_extensive_sweep(self):
        state = fields()
        speed = 0.27
        measure = 1.4
        result = moving_separation_ibl_flux(
            state,
            outward_surface_conormal=[[1.0, 0.0, 0.0]],
            relative_conormal_speed=speed,
            edge_measure=measure,
        )
        np.testing.assert_allclose(
            result.boundary_motion_momentum_defect_out,
            -speed*measure*state.mass_flux_defect,
            atol=0.0,
        )
        np.testing.assert_allclose(
            result.boundary_motion_energy_defect_out,
            -speed*measure*state.momentum_flux_trace,
            atol=0.0,
        )
        np.testing.assert_allclose(
            result.relative_momentum_defect_out,
            result.physical_momentum_defect_out
            + result.boundary_motion_momentum_defect_out,
            atol=0.0,
        )

    def test_orientation_and_speed_reversal_negates_flux(self):
        state = fields()
        reference = moving_separation_ibl_flux(
            state,
            outward_surface_conormal=[[1.0, 0.0, 0.0]],
            relative_conormal_speed=-0.19,
            edge_measure=0.8,
        )
        reversed_result = moving_separation_ibl_flux(
            state,
            outward_surface_conormal=[[-1.0, 0.0, 0.0]],
            relative_conormal_speed=0.19,
            edge_measure=0.8,
        )
        np.testing.assert_allclose(
            reversed_result.relative_momentum_defect_out,
            -reference.relative_momentum_defect_out,
            atol=2.0e-16,
        )
        np.testing.assert_allclose(
            reversed_result.relative_energy_defect_out,
            -reference.relative_energy_defect_out,
            atol=2.0e-16,
        )

    def test_invalid_speed_measure_and_conormal_fail(self):
        state = fields()
        valid = dict(
            fields=state,
            outward_surface_conormal=[[1.0, 0.0, 0.0]],
            relative_conormal_speed=0.0,
            edge_measure=1.0,
        )
        for replacement in (
            {"relative_conormal_speed": np.nan},
            {"relative_conormal_speed": [0.0, 0.1]},
            {"edge_measure": 0.0},
            {"outward_surface_conormal": [[1.0, 0.0, 0.1]]},
        ):
            arguments = dict(valid)
            arguments.update(replacement)
            with self.assertRaises(SurfaceIBLError):
                moving_separation_ibl_flux(**arguments)


if __name__ == "__main__":
    unittest.main()

