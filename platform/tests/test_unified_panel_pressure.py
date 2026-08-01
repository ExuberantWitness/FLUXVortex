import sys
import unittest
from pathlib import Path

import numpy as np

PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from claim_runtime.unified_panel_pressure import (  # noqa: E402
    UnifiedPressureError,
    structured_uvlm_surface_gradient,
    unified_panel_pressure,
)
from claim_runtime.work_conjugate_transfer import (  # noqa: E402
    rigid_body_jacobian,
    rigid_resultant_report,
    transfer_generalized,
)


class UnifiedPressureTests(unittest.TestCase):
    def setUp(self):
        self.rho = 1.225
        self.nc, self.ns = 3, 4
        self.count = self.nc * self.ns
        rng = np.random.default_rng(31)
        self.gamma = rng.normal(size=self.count)
        self.gamma_previous = rng.normal(size=self.count)
        self.dt = 0.017
        self.tc = np.tile([1.0, 0.0, 0.0], (self.count, 1))
        self.ts = np.tile([0.0, 1.0, 0.0], (self.count, 1))
        self.dx = rng.uniform(0.08, 0.15, size=self.count)
        self.dy = rng.uniform(0.06, 0.13, size=self.count)
        self.velocity = rng.normal(size=(self.count, 3))
        self.normal = np.tile([0.0, 0.0, 1.0], (self.count, 1))
        self.area = rng.uniform(0.002, 0.01, size=self.count)

    def test_matches_frozen_n1_structured_formula_panel_by_panel(self):
        gradient = structured_uvlm_surface_gradient(
            self.gamma,
            chord_tangent=self.tc,
            span_tangent=self.ts,
            chord_step=self.dx,
            span_step=self.dy,
            nc=self.nc,
            ns=self.ns,
        )
        rate = (self.gamma - self.gamma_previous) / self.dt
        result = unified_panel_pressure(
            density=self.rho,
            local_velocity=self.velocity,
            surface_gradient=gradient,
            potential_rate_channels={"bound_unsteady": rate},
            area=self.area,
            normal=self.normal,
        )

        gamma = self.gamma.reshape(self.nc, self.ns)
        dx = self.dx.reshape(self.nc, self.ns)
        dy = self.dy.reshape(self.nc, self.ns)
        dgamma_dx = np.empty_like(gamma)
        dgamma_dx[0] = gamma[0] / dx[0]
        dgamma_dx[1:] = (gamma[1:] - gamma[:-1]) / dx[1:]
        dgamma_dy = np.zeros_like(gamma)
        dgamma_dy[:, 0] = gamma[:, 0] / dy[:, 0]
        dgamma_dy[:, -1] = -gamma[:, -1] / dy[:, -1]
        dgamma_dy[:, 1:-1] = (
            gamma[:, 2:] - gamma[:, :-2]
        ) / (2.0 * dy[:, 1:-1])
        expected = self.rho * (
            self.velocity[:, 0] * dgamma_dx.ravel()
            + self.velocity[:, 1] * dgamma_dy.ravel()
            + rate
        )
        np.testing.assert_allclose(result.total_pressure, expected, atol=2e-14)
        np.testing.assert_allclose(
            result.total_force,
            expected[:, None] * self.area[:, None] * self.normal,
            atol=2e-14,
        )
        self.assertTrue(result.ledger_report().passed)

    def test_hirato_eq17_rate_channels_recompose_before_force(self):
        gradient = np.zeros((self.count, 3))
        bound_rate = np.linspace(-0.4, 0.5, self.count)
        lev_rate = np.linspace(0.2, -0.1, self.count)
        result = unified_panel_pressure(
            density=self.rho,
            local_velocity=self.velocity,
            surface_gradient=gradient,
            potential_rate_channels={
                "bound_unsteady": bound_rate,
                "lev_sheet_unsteady": lev_rate,
            },
            area=self.area,
            normal=self.normal,
        )
        np.testing.assert_allclose(
            result.total_pressure,
            self.rho * (bound_rate + lev_rate),
            atol=2e-15,
        )
        self.assertEqual(
            set(result.pressure_channels),
            {"surface_advection", "bound_unsteady", "lev_sheet_unsteady"},
        )

    def test_pressure_points_transfer_with_force_moment_and_work_conservation(self):
        rng = np.random.default_rng(9)
        points = rng.normal(size=(self.count, 3))
        result = unified_panel_pressure(
            density=self.rho,
            local_velocity=self.velocity,
            surface_gradient=np.zeros((self.count, 3)),
            potential_rate_channels={
                "bound_unsteady": np.linspace(-1.0, 1.0, self.count)
            },
            area=self.area,
            normal=self.normal,
        )
        origin = np.array([0.1, -0.2, 0.05])
        jacobian = rigid_body_jacobian(points, origin=origin)
        load = transfer_generalized(result.total_force, jacobian)
        self.assertTrue(
            rigid_resultant_report(
                points,
                result.total_force,
                load.values,
                origin=origin,
            ).passed
        )
        probes = rng.normal(size=(7, 6))
        self.assertTrue(load.virtual_work_report(probes).passed)

    def test_reserved_channel_and_nonunit_normal_fail(self):
        with self.assertRaises(UnifiedPressureError):
            unified_panel_pressure(
                density=self.rho,
                local_velocity=self.velocity,
                surface_gradient=np.zeros((self.count, 3)),
                potential_rate_channels={
                    "surface_advection": np.zeros(self.count)
                },
                area=self.area,
                normal=self.normal,
            )
        with self.assertRaises(UnifiedPressureError):
            unified_panel_pressure(
                density=self.rho,
                local_velocity=self.velocity,
                surface_gradient=np.zeros((self.count, 3)),
                potential_rate_channels={"bound": np.zeros(self.count)},
                area=self.area,
                normal=2.0 * self.normal,
            )


if __name__ == "__main__":
    unittest.main()
