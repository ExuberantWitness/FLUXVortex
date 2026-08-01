import sys
import unittest
from pathlib import Path

import numpy as np

PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from claim_runtime.distributed_doublet import (  # noqa: E402
    QuadraticDoubletElement,
)
from claim_runtime.material_helmholtz import (  # noqa: E402
    material_p2_helmholtz_report,
)


def material_nodes(vertices):
    v0, v1, v2 = vertices
    return np.stack(
        (v0, v1, v2, 0.5 * (v0 + v1), 0.5 * (v1 + v2), 0.5 * (v2 + v0))
    )


class MaterialHelmholtzTests(unittest.TestCase):
    def test_material_p2_scalar_generates_exact_cauchy_stretching(self):
        reference_vertices = np.array(
            [[0.0, 0.0, 0.0], [1.2, 0.1, 0.0], [0.2, 0.9, 0.0]]
        )
        reference_nodes = material_nodes(reference_vertices)
        xi = np.array([0.0, 1.0, 0.0, 0.5, 0.5, 0.0])
        eta = np.array([0.0, 0.0, 1.0, 0.0, 0.5, 0.5])
        mu = (
            0.2
            + 0.7 * xi
            - 0.4 * eta
            + 0.3 * xi**2
            + 0.2 * xi * eta
            - 0.1 * eta**2
        )
        reference = QuadraticDoubletElement(reference_vertices, mu)
        current_vertices = np.array(
            [[0.3, -0.2, 0.1], [1.7, 0.25, 0.45], [-0.1, 1.2, 0.7]]
        )
        current = reference.material_update(current_vertices)
        barycentric = np.array(
            [
                [0.2, 0.3, 0.5],
                [0.6, 0.1, 0.3],
                [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            ]
        )
        report = material_p2_helmholtz_report(
            reference,
            current,
            barycentric,
            tolerance=2.0e-14,
        )
        self.assertTrue(report.passed, report)
        self.assertLess(report.max_material_mu_residual, 2.0e-15)
        self.assertLess(
            report.max_cauchy_vector_density_residual,
            2.0e-14,
        )

    def test_changing_material_mu_fails_kelvin_identity(self):
        reference = QuadraticDoubletElement(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            np.arange(6, dtype=float) * 0.1,
        )
        current = QuadraticDoubletElement(
            [[0.0, 0.0, 0.0], [1.2, 0.1, 0.0], [0.0, 0.8, 0.2]],
            reference.material_mu + np.array([0.0, 0.0, 0.1, 0.0, 0.0, 0.0]),
        )
        report = material_p2_helmholtz_report(
            reference,
            current,
            [[0.2, 0.2, 0.6]],
        )
        self.assertFalse(report.passed)
        self.assertGreater(report.max_material_mu_residual, 0.0)


if __name__ == "__main__":
    unittest.main()
