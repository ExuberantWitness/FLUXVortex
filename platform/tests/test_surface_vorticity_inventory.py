import sys
import unittest
from pathlib import Path

import numpy as np

PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from claim_runtime.surface_vorticity_inventory import (  # noqa: E402
    SurfaceInventoryError,
    surface_inventory_budget_report,
)


class SurfaceInventoryTests(unittest.TestCase):
    def manufactured(self, perturb=0.0):
        previous = np.array(
            [[0.2, -0.1, 0.0], [0.0, 0.3, 0.2], [-0.2, 0.1, 0.4]]
        )
        wall = np.array(
            [[0.4, 0.1, -0.2], [0.2, -0.3, 0.1], [-0.1, 0.2, 0.3]]
        )
        edges = np.array([[0, 1], [1, 2]])
        flux = np.array([[0.1, -0.2, 0.05], [-0.3, 0.1, 0.2]])
        internal = np.zeros_like(previous)
        for (source, destination), value in zip(edges, flux):
            internal[source] -= value
            internal[destination] += value
        external = np.array(
            [[0.0, 0.1, 0.0], [0.0, 0.0, 0.0], [-0.05, 0.0, 0.1]]
        )
        release = np.array(
            [[0.0, 0.0, 0.0], [0.15, -0.05, 0.0], [0.1, 0.0, 0.2]]
        )
        dt = 0.02
        current = previous+dt*(wall+internal+external-release)
        current[1, 0] += perturb
        return surface_inventory_budget_report(
            previous_inventory=previous,
            current_inventory=current,
            dt=dt,
            wall_transfer_rate=wall,
            internal_edges=edges,
            internal_edge_flux_rate=flux,
            external_transport_net_in_rate=external,
            separation_release_rate=release,
            tolerance=1.0e-13,
        )

    def test_manufactured_budget_and_internal_cancellation(self):
        report = self.manufactured()
        self.assertTrue(report.passed)
        self.assertLessEqual(
            report.global_internal_flux_residual, 1.0e-15
        )
        self.assertLessEqual(report.max_local_residual, 1.0e-13)

    def test_ledger_reports_but_does_not_absorb_missing_release(self):
        report = self.manufactured(perturb=1.0e-4)
        self.assertFalse(report.passed)
        self.assertGreater(report.max_local_residual, 1.0e-3)

    def test_bad_edge_and_missing_release_shape_fail(self):
        with self.assertRaises(SurfaceInventoryError):
            surface_inventory_budget_report(
                previous_inventory=np.zeros((2, 3)),
                current_inventory=np.zeros((2, 3)),
                dt=0.1,
                wall_transfer_rate=np.zeros((2, 3)),
                internal_edges=[[0, 2]],
                internal_edge_flux_rate=np.zeros((1, 3)),
                external_transport_net_in_rate=np.zeros((2, 3)),
                separation_release_rate=np.zeros((2, 3)),
            )
        with self.assertRaises(SurfaceInventoryError):
            surface_inventory_budget_report(
                previous_inventory=np.zeros((2, 3)),
                current_inventory=np.zeros((2, 3)),
                dt=0.1,
                wall_transfer_rate=np.zeros((2, 3)),
                internal_edges=[],
                internal_edge_flux_rate=np.zeros((0, 3)),
                external_transport_net_in_rate=np.zeros((2, 3)),
                separation_release_rate=np.zeros((1, 3)),
            )


if __name__ == "__main__":
    unittest.main()
