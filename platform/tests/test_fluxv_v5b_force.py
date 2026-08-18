from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from claim_runtime.hirato_equations import potential_rate_eq17  # noqa: E402
from claim_runtime.hirato_live_shadow import (  # noqa: E402
    HiratoLiveShadow,
    build_bound_lattice,
)
from claim_runtime.unified_panel_pressure import (  # noqa: E402
    structured_uvlm_surface_gradient,
)
from forward_flight_benchmarks.fluxv_v5b_force import (  # noqa: E402
    FluxVV5BForceError,
    fluxv_v5b_surface_force,
    n1_no_lev_pressure_baseline,
    reconstruct_panel_surface_geometry,
)


def rectangular_wing(nc: int, ns: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0.0, 1.0, nc + 1)
    y = np.linspace(0.0, 1.5, ns + 1)
    corners = np.zeros((nc + 1, ns + 1, 3))
    corners[..., 0] = x[:, None]
    corners[..., 1] = y[None, :]
    return corners, np.zeros_like(corners)


def pitched_wing(
    nc: int,
    ns: int,
    alpha_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    corners, velocity = rectangular_wing(nc, ns)
    angle = np.deg2rad(alpha_deg)
    relative = corners[..., 0] - 0.25
    corners[..., 0] = 0.25 + relative * np.cos(angle)
    corners[..., 2] = relative * np.sin(angle)
    return corners, velocity


class FluxVV5BForceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.nc = 2
        self.ns = 3
        self.u_inf = np.array([2.0, 0.0, 0.0])
        self.dt = 0.01
        self.rho = 1.225
        self.core = 0.01

    def _state(
        self,
        corners: np.ndarray,
        velocity: np.ndarray,
        *,
        lesp_crit: float,
    ):
        shadow = HiratoLiveShadow(
            nc=self.nc,
            ns=self.ns,
            u_infinity=self.u_inf,
            dt=self.dt,
            lesp_crit=lesp_crit,
            core_radius=self.core,
            mirror_symmetry=False,
        )
        report = shadow.step(
            step=0,
            corners=corners,
            corner_velocity=velocity,
        )
        lattice = build_bound_lattice(
            corners,
            velocity,
            nc=self.nc,
            ns=self.ns,
        )
        return shadow, lattice, report

    def test_no_lev_is_exact_same_n1_pressure_path(self) -> None:
        corners, velocity = rectangular_wing(self.nc, self.ns)
        _, lattice, report = self._state(
            corners,
            velocity,
            lesp_crit=10.0,
        )
        previous = np.zeros(self.nc * self.ns)
        baseline = n1_no_lev_pressure_baseline(
            density=self.rho,
            dt=self.dt,
            nc=self.nc,
            ns=self.ns,
            lattice=lattice,
            report=report,
            previous_bound_gamma=previous,
            u_infinity=self.u_inf,
            core_radius=self.core,
            mirror_symmetry=False,
        )
        v5b = fluxv_v5b_surface_force(
            density=self.rho,
            dt=self.dt,
            nc=self.nc,
            ns=self.ns,
            lattice=lattice,
            report=report,
            previous_bound_gamma=previous,
            previous_gamma_lev=np.zeros(self.ns),
            u_infinity=self.u_inf,
            core_radius=self.core,
            mirror_symmetry=False,
        )

        for actual, expected in (
            (v5b.pressure.total_pressure, baseline.pressure.total_pressure),
            (v5b.panel_force, baseline.panel_force),
            (v5b.panel_moment, baseline.panel_moment),
            (v5b.total_force, baseline.total_force),
            (v5b.total_moment, baseline.total_moment),
            (v5b.local_velocity, baseline.local_velocity),
            (v5b.surface_gradient, baseline.surface_gradient),
        ):
            self.assertTrue(np.array_equal(actual, expected))
        self.assertTrue(v5b.guards.no_lev_exact_reduction_required)
        self.assertTrue(v5b.guards.no_lev_exact_reduction_passed)
        self.assertTrue(v5b.guards.passed)

    def test_active_step_has_one_velocity_pressure_and_force_ledger(self) -> None:
        corners, velocity = pitched_wing(self.nc, self.ns, 15.0)
        velocity[..., 0] = 0.15
        velocity[..., 2] = -0.04
        _, lattice, report = self._state(
            corners,
            velocity,
            lesp_crit=0.05,
        )
        self.assertTrue(np.any(report.active))
        previous_bound = np.zeros(self.nc * self.ns)
        previous_lev = np.zeros(self.ns)
        ledger = fluxv_v5b_surface_force(
            density=self.rho,
            dt=self.dt,
            nc=self.nc,
            ns=self.ns,
            lattice=lattice,
            report=report,
            previous_bound_gamma=previous_bound,
            previous_gamma_lev=previous_lev,
            u_infinity=self.u_inf,
            core_radius=self.core,
            mirror_symmetry=False,
            moment_origin=np.array([0.25, 0.0, 0.0]),
        )

        self.assertEqual(
            set(ledger.local_velocity_channels),
            {"freestream_motion", "bound", "tev", "lev"},
        )
        np.testing.assert_allclose(
            ledger.local_velocity_channels["freestream_motion"],
            self.u_inf[None] - lattice.collocation_velocity,
            atol=0.0,
        )
        np.testing.assert_allclose(
            ledger.local_velocity,
            np.sum(
                np.stack(tuple(ledger.local_velocity_channels.values())),
                axis=0,
            ),
            atol=2e-14,
        )
        expected_rate = potential_rate_eq17(
            report.bound_gamma.reshape(self.nc, self.ns),
            previous_bound.reshape(self.nc, self.ns),
            report.gamma_lev,
            previous_lev,
            report.active,
            self.dt,
        )
        np.testing.assert_allclose(ledger.potential_rate.bound, expected_rate.bound)
        np.testing.assert_allclose(ledger.potential_rate.lev, expected_rate.lev)
        np.testing.assert_allclose(
            ledger.pressure.pressure_channels["bound_unsteady"],
            self.rho * expected_rate.bound.ravel(),
        )
        np.testing.assert_allclose(
            ledger.pressure.pressure_channels["lev_sheet_unsteady"],
            self.rho * expected_rate.lev.ravel(),
        )

        expected_jump = report.bound_gamma + np.tile(
            np.where(report.active, report.gamma_lev, 0.0),
            self.nc,
        )
        geometry = reconstruct_panel_surface_geometry(
            lattice,
            nc=self.nc,
            ns=self.ns,
        )
        expected_gradient = structured_uvlm_surface_gradient(
            expected_jump,
            chord_tangent=geometry.chord_tangent,
            span_tangent=geometry.span_tangent,
            chord_step=geometry.chord_step,
            span_step=geometry.span_step,
            nc=self.nc,
            ns=self.ns,
        )
        np.testing.assert_allclose(ledger.surface_potential_jump, expected_jump)
        np.testing.assert_allclose(ledger.surface_gradient, expected_gradient)
        np.testing.assert_allclose(
            ledger.total_force,
            np.sum(ledger.panel_force, axis=0),
        )
        np.testing.assert_allclose(
            ledger.total_moment,
            np.sum(ledger.panel_moment, axis=0),
        )
        self.assertEqual(
            set(ledger.pressure.pressure_channels),
            {"surface_advection", "bound_unsteady", "lev_sheet_unsteady"},
        )
        self.assertTrue(ledger.guards.passed)

    def test_report_lattice_mismatch_and_hidden_lev_baseline_fail_closed(self) -> None:
        corners, velocity = pitched_wing(self.nc, self.ns, 15.0)
        _, lattice, report = self._state(
            corners,
            velocity,
            lesp_crit=0.05,
        )
        with self.assertRaises(FluxVV5BForceError):
            n1_no_lev_pressure_baseline(
                density=self.rho,
                dt=self.dt,
                nc=self.nc,
                ns=self.ns,
                lattice=lattice,
                report=report,
                previous_bound_gamma=np.zeros(self.nc * self.ns),
                u_infinity=self.u_inf,
                core_radius=self.core,
                mirror_symmetry=False,
            )

        other_corners = corners.copy()
        other_corners[-1, -1, 2] += 1.0e-5
        other_lattice = build_bound_lattice(
            other_corners,
            velocity,
            nc=self.nc,
            ns=self.ns,
        )
        with self.assertRaises(FluxVV5BForceError):
            fluxv_v5b_surface_force(
                density=self.rho,
                dt=self.dt,
                nc=self.nc,
                ns=self.ns,
                lattice=other_lattice,
                report=report,
                previous_bound_gamma=np.zeros(self.nc * self.ns),
                previous_gamma_lev=np.zeros(self.ns),
                u_infinity=self.u_inf,
                core_radius=self.core,
                mirror_symmetry=False,
            )


if __name__ == "__main__":
    unittest.main()
