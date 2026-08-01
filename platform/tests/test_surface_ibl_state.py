import sys
import unittest
from pathlib import Path

import numpy as np

PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from claim_runtime.surface_ibl_state import (  # noqa: E402
    SurfaceIBLError,
    SurfaceIBLFields,
    rotate_surface_ibl_fields,
    surface_ibl_budget_report,
    surface_ibl_physical_flux,
    surface_ibl_source_terms,
)


def fields(mass, trace, *, tensor_xy=None):
    mass = np.asarray(mass, dtype=float)
    trace = np.asarray(trace, dtype=float)
    count = len(mass)
    tensor = np.zeros((count, 3, 3))
    tensor[:, 0, 0] = trace
    if tensor_xy is not None:
        tensor[:, 0, 1] = tensor_xy
    return SurfaceIBLFields(
        mass_flux_defect=mass,
        momentum_flux_defect=tensor,
        kinetic_energy_defect_flux=np.zeros((count, 3)),
        external_tangential_velocity=np.zeros((count, 3)),
        external_velocity_surface_gradient=np.zeros((count, 3, 3)),
        wall_shear_over_density=np.zeros((count, 3)),
        dissipation_integral=np.zeros(count),
        surface_normal=np.tile([0.0, 0.0, 1.0], (count, 1)),
    )


class SurfaceIBLStateTests(unittest.TestCase):
    def test_planar_two_cell_manufactured_balance(self):
        dt = 0.2
        previous = fields(
            [[0.4, -0.1, 0.0], [0.2, 0.3, 0.0]],
            [0.5, 0.7],
        )
        momentum_flux = np.array([[0.12, -0.04, 0.0]])
        energy_flux = np.array([0.09])
        internal_momentum = np.array([
            -momentum_flux[0],
            momentum_flux[0],
        ])
        internal_energy = np.array([-energy_flux[0], energy_flux[0]])
        boundary_momentum = np.array([
            [0.03, 0.02, 0.0],
            [-0.01, 0.04, 0.0],
        ])
        boundary_energy = np.array([0.02, -0.03])
        source_momentum = np.array([
            [0.01, -0.02, 0.0],
            [0.05, 0.01, 0.0],
        ])
        source_energy = np.array([0.04, 0.01])
        momentum_rate = (
            internal_momentum+boundary_momentum+source_momentum
        )
        energy_rate = internal_energy+boundary_energy+source_energy
        current = fields(
            previous.mass_flux_defect+dt*momentum_rate,
            previous.momentum_flux_trace+dt*energy_rate,
        )
        report = surface_ibl_budget_report(
            previous_fields=previous,
            current_fields=current,
            previous_cell_area=np.ones(2),
            current_cell_area=np.ones(2),
            dt=dt,
            internal_edges=[[0, 1]],
            internal_momentum_flux_out_of_first_rate=momentum_flux,
            internal_energy_flux_out_of_first_rate=energy_flux,
            boundary_momentum_net_in_rate=boundary_momentum,
            boundary_energy_net_in_rate=boundary_energy,
            momentum_source_integral_rate=source_momentum,
            energy_source_integral_rate=source_energy,
        )
        self.assertTrue(report.passed)
        self.assertLessEqual(report.max_momentum_residual, 1.0e-15)
        self.assertLessEqual(report.max_energy_residual, 1.0e-15)
        self.assertEqual(report.global_internal_momentum_flux_residual, 0.0)
        self.assertEqual(report.global_internal_energy_flux_residual, 0.0)

    def test_moving_area_uses_extensive_storage(self):
        dt = 0.1
        previous_area = np.array([1.0])
        current_area = np.array([1.25])
        previous = fields([[0.4, -0.2, 0.0]], [0.6])
        momentum_source = np.array([[0.15, 0.05, 0.0]])
        energy_source = np.array([0.08])
        current_mass = (
            previous_area[:, None]*previous.mass_flux_defect
            + dt*momentum_source
        )/current_area[:, None]
        current_trace = (
            previous_area*previous.momentum_flux_trace+dt*energy_source
        )/current_area
        current = fields(current_mass, current_trace)
        report = surface_ibl_budget_report(
            previous_fields=previous,
            current_fields=current,
            previous_cell_area=previous_area,
            current_cell_area=current_area,
            dt=dt,
            internal_edges=np.empty((0, 2), dtype=int),
            internal_momentum_flux_out_of_first_rate=np.empty((0, 3)),
            internal_energy_flux_out_of_first_rate=np.empty(0),
            boundary_momentum_net_in_rate=np.zeros((1, 3)),
            boundary_energy_net_in_rate=np.zeros(1),
            momentum_source_integral_rate=momentum_source,
            energy_source_integral_rate=energy_source,
        )
        self.assertTrue(report.passed)
        self.assertLessEqual(report.max_momentum_residual, 1.0e-15)
        self.assertLessEqual(report.max_energy_residual, 1.0e-15)

    def test_source_terms_are_objective_under_proper_rotation(self):
        tensor = np.array([[
            [0.6, -0.2, 0.0],
            [0.1, 0.4, 0.0],
            [0.0, 0.0, 0.0],
        ]])
        gradient = np.array([[
            [0.2, 0.3, 0.0],
            [-0.4, 0.1, 0.0],
            [0.0, 0.0, 0.0],
        ]])
        original = SurfaceIBLFields(
            mass_flux_defect=[[0.3, -0.1, 0.0]],
            momentum_flux_defect=tensor,
            kinetic_energy_defect_flux=[[0.8, 0.2, 0.0]],
            external_tangential_velocity=[[2.0, -0.5, 0.0]],
            external_velocity_surface_gradient=gradient,
            wall_shear_over_density=[[0.02, -0.03, 0.0]],
            dissipation_integral=[0.07],
            surface_normal=[[0.0, 0.0, 1.0]],
        )
        angle = 0.71
        axis = np.array([1.0, 2.0, -0.5])
        axis /= np.linalg.norm(axis)
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
        reference = surface_ibl_source_terms(original)
        moved = rotate_surface_ibl_fields(original, rotation)
        transformed = surface_ibl_source_terms(moved)
        expected_momentum = np.einsum(
            "ij,nj->ni",
            rotation,
            reference.momentum,
        )
        np.testing.assert_allclose(
            transformed.momentum,
            expected_momentum,
            atol=2.0e-15,
        )
        np.testing.assert_allclose(
            transformed.energy,
            reference.energy,
            atol=2.0e-15,
        )
        np.testing.assert_allclose(
            moved.momentum_flux_trace,
            original.momentum_flux_trace,
            atol=2.0e-15,
        )

    def test_one_dimensional_reduction_matches_scalar_equations(self):
        mass = 0.35
        tensor = 0.42
        energy_flux = 0.73
        velocity = 1.8
        gradient = -0.16
        shear = 0.025
        dissipation = 0.031
        state = SurfaceIBLFields(
            mass_flux_defect=[[mass, 0.0, 0.0]],
            momentum_flux_defect=[[
                [tensor, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]],
            kinetic_energy_defect_flux=[[energy_flux, 0.0, 0.0]],
            external_tangential_velocity=[[velocity, 0.0, 0.0]],
            external_velocity_surface_gradient=[[
                [gradient, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]],
            wall_shear_over_density=[[shear, 0.0, 0.0]],
            dissipation_integral=[dissipation],
            surface_normal=[[0.0, 0.0, 1.0]],
        )
        source = surface_ibl_source_terms(state)
        flux = surface_ibl_physical_flux(
            state,
            outward_surface_conormal=[[1.0, 0.0, 0.0]],
        )
        expected_momentum_source = -gradient*mass+shear
        expected_energy_source = (
            2.0*dissipation-tensor*gradient
            + velocity*(gradient*mass-shear)
        )
        self.assertAlmostEqual(
            source.momentum[0, 0],
            expected_momentum_source,
        )
        np.testing.assert_allclose(source.momentum[0, 1:], 0.0, atol=0.0)
        self.assertAlmostEqual(source.energy[0], expected_energy_source)
        self.assertAlmostEqual(flux.momentum_out[0, 0], tensor)
        self.assertAlmostEqual(
            flux.energy_out[0],
            energy_flux-tensor*velocity,
        )

    def test_omitted_source_remains_visible_and_bad_tangency_fails(self):
        previous = fields([[0.0, 0.0, 0.0]], [0.0])
        current = fields([[0.01, 0.0, 0.0]], [0.02])
        report = surface_ibl_budget_report(
            previous_fields=previous,
            current_fields=current,
            previous_cell_area=[1.0],
            current_cell_area=[1.0],
            dt=0.1,
            internal_edges=np.empty((0, 2), dtype=int),
            internal_momentum_flux_out_of_first_rate=np.empty((0, 3)),
            internal_energy_flux_out_of_first_rate=np.empty(0),
            boundary_momentum_net_in_rate=np.zeros((1, 3)),
            boundary_energy_net_in_rate=np.zeros(1),
            momentum_source_integral_rate=np.zeros((1, 3)),
            energy_source_integral_rate=np.zeros(1),
        )
        self.assertFalse(report.passed)
        self.assertGreater(report.max_momentum_residual, 0.099)
        self.assertGreater(report.max_energy_residual, 0.199)
        with self.assertRaises(SurfaceIBLError):
            SurfaceIBLFields(
                mass_flux_defect=[[0.0, 0.0, 1.0]],
                momentum_flux_defect=np.zeros((1, 3, 3)),
                kinetic_energy_defect_flux=np.zeros((1, 3)),
                external_tangential_velocity=np.zeros((1, 3)),
                external_velocity_surface_gradient=np.zeros((1, 3, 3)),
                wall_shear_over_density=np.zeros((1, 3)),
                dissipation_integral=np.zeros(1),
                surface_normal=[[0.0, 0.0, 1.0]],
            )


if __name__ == "__main__":
    unittest.main()
