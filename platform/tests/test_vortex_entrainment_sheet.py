import sys
import unittest
from pathlib import Path

import numpy as np

PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from claim_runtime.vortex_entrainment_sheet import (  # noqa: E402
    VortexEntrainmentError,
    equal_density_mass_flux_jump,
    release_junction_report,
    vortex_entrainment_local_balance,
    vortex_entrainment_velocity,
)


def _square_quadrature(count=40):
    coordinate = (np.arange(count)+0.5)/count-0.5
    x, y = np.meshgrid(coordinate, coordinate, indexing="ij")
    points = np.stack((x.ravel(), y.ravel(), np.zeros(x.size)), axis=1)
    area = np.full(x.size, 1.0/x.size)
    normal = np.tile([0.0, 0.0, 1.0], (x.size, 1))
    return points, area, normal


class VortexEntrainmentSheetTests(unittest.TestCase):
    def test_vortex_and_entrainment_are_independent_and_linear(self):
        source, area, normal = _square_quadrature()
        gamma = np.tile([0.0, 0.7, 0.0], (len(source), 1))
        q = np.full(len(source), 0.35)
        target = np.array([[0.0, 0.0, 1.7]])
        combined = vortex_entrainment_velocity(
            target, source, area, gamma, q, normal
        )
        vortex_only = vortex_entrainment_velocity(
            target, source, area, gamma, np.zeros_like(q), normal
        )
        entrainment_only = vortex_entrainment_velocity(
            target, source, area, np.zeros_like(gamma), q, normal
        )
        np.testing.assert_allclose(
            combined.velocity,
            vortex_only.velocity+entrainment_only.velocity,
            atol=1e-15,
        )
        self.assertGreater(abs(vortex_only.velocity[0, 0]), 1e-4)
        self.assertGreater(abs(entrainment_only.velocity[0, 2]), 1e-4)
        self.assertLess(np.max(np.abs(vortex_only.velocity[0, 1:])), 1e-15)
        self.assertLess(np.max(np.abs(
            entrainment_only.velocity[0, :2]
        )), 1e-15)

    def test_rotation_covariance(self):
        source, area, normal = _square_quadrature(24)
        gamma = np.tile([0.2, -0.4, 0.0], (len(source), 1))
        q = np.linspace(0.1, 0.3, len(source))
        target = np.array([[0.2, -0.1, 1.2], [-0.3, 0.2, 2.1]])
        reference = vortex_entrainment_velocity(
            target, source, area, gamma, q, normal
        )
        axis = np.array([0.3, -0.8, 0.5])
        axis /= np.linalg.norm(axis)
        angle = 0.91
        skew = np.array([
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ])
        rotation = (
            np.eye(3)*np.cos(angle)
            +(1.0-np.cos(angle))*np.outer(axis, axis)
            +np.sin(angle)*skew
        )
        rotated = vortex_entrainment_velocity(
            target@rotation.T,
            source@rotation.T,
            area,
            gamma@rotation.T,
            q,
            normal@rotation.T,
        )
        np.testing.assert_allclose(
            rotated.velocity,
            reference.velocity@rotation.T,
            atol=2e-15,
        )

    def test_local_balances_and_release_ledger(self):
        rho_s = 0.42
        divergence = -0.17
        bulk_density = 1.225
        q = 0.31
        mass_jump = float(equal_density_mass_flux_jump(
            bulk_density,
            np.array(q),
        ))
        drho_s = -(rho_s*divergence+mass_jump)
        acceleration = np.array([0.3, -0.4, 0.2])
        stress_divergence = np.array([0.02, 0.03, -0.01])
        body_force = np.array([0.0, 0.0, -0.2])
        momentum_jump = np.array([0.04, -0.06, 0.08])
        pressure_jump = 0.15
        normal = np.array([0.0, 0.0, 1.0])
        shear_jump = (
            rho_s*acceleration
            -(stress_divergence+rho_s*body_force)
            +momentum_jump
            +pressure_jump*normal
        )
        balance = vortex_entrainment_local_balance(
            surface_mass_density=rho_s,
            material_surface_mass_rate=drho_s,
            surface_velocity_divergence=divergence,
            outer_mass_flux_jump=mass_jump,
            material_surface_acceleration=acceleration,
            surface_stress_divergence=stress_divergence,
            surface_body_force=body_force,
            outer_momentum_flux_jump=momentum_jump,
            pressure_jump=pressure_jump,
            normal=normal,
            shear_stress_jump=shear_jump,
        )
        self.assertLess(balance.maximum_absolute_residual, 2e-16)

        circulation = np.array([
            [0.1, -0.2, 0.05],
            [-0.03, 0.04, 0.02],
        ])
        mass = np.array([0.03, 0.04])
        momentum = np.array([
            [0.2, -0.1, 0.3],
            [-0.05, 0.07, 0.02],
        ])
        entrainment = np.array([0.012, 0.018])
        report = release_junction_report(
            incoming_circulation_rate=circulation,
            newborn_circulation_rate=circulation.sum(axis=0),
            incoming_mass_rate=mass,
            newborn_mass_rate=mass.sum(),
            incoming_momentum_rate=momentum,
            newborn_momentum_rate=momentum.sum(axis=0),
            incoming_entrainment_rate=entrainment,
            newborn_entrainment_rate=entrainment.sum(),
        )
        self.assertLess(report.maximum_absolute_residual, 1e-15)

    def test_singular_target_and_nontangent_gamma_fail(self):
        source, area, normal = _square_quadrature(2)
        with self.assertRaises(VortexEntrainmentError):
            vortex_entrainment_velocity(
                source[:1],
                source,
                area,
                np.zeros_like(source),
                np.zeros(len(source)),
                normal,
            )
        bad_gamma = np.tile([0.0, 0.0, 1.0], (len(source), 1))
        with self.assertRaises(VortexEntrainmentError):
            vortex_entrainment_velocity(
                np.array([[0.0, 0.0, 1.0]]),
                source,
                area,
                bad_gamma,
                np.zeros(len(source)),
                normal,
            )


if __name__ == "__main__":
    unittest.main()

