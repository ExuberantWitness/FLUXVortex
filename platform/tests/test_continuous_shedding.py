import sys
import unittest
from pathlib import Path

import numpy as np

PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from claim_runtime.continuous_shedding import (  # noqa: E402
    newborn_halfwing_shedding_band,
    reconstruct_halfwing_p2_trace,
)
from claim_runtime.distributed_doublet import (  # noqa: E402
    DistributedDoubletError,
)


class ContinuousSheddingTests(unittest.TestCase):
    def test_manufactured_quadratic_trace_is_recovered_exactly(self):
        edges = np.linspace(0.0, 1.0, 9)
        midpoints = 0.5 * (edges[:-1] + edges[1:])
        reconstruction = reconstruct_halfwing_p2_trace(
            1.0 - midpoints**2,
            edges,
        )
        expected = 1.0 - reconstruction.span_p2_coordinates**2
        np.testing.assert_allclose(
            reconstruction.p2_values,
            expected,
            atol=2.0e-14,
        )
        query = np.linspace(0.0, 1.0, 37)
        np.testing.assert_allclose(
            reconstruction.evaluate(query),
            1.0 - query**2,
            atol=2.0e-14,
        )
        self.assertTrue(reconstruction.report.passed)

    def test_nonuniform_grid_preserves_registered_constraints(self):
        edges = np.linspace(0.0, 1.0, 12) ** 1.3
        midpoints = 0.5 * (edges[:-1] + edges[1:])
        reconstruction = reconstruct_halfwing_p2_trace(
            np.cos(0.5 * np.pi * midpoints),
            edges,
        )
        report = reconstruction.report
        self.assertTrue(report.passed)
        self.assertLess(report.max_midpoint_residual, 1.0e-12)
        self.assertLess(report.root_derivative_residual, 1.0e-11)
        self.assertLess(report.tip_value_residual, 1.0e-12)
        self.assertLess(report.max_internal_derivative_jump, 1.0e-11)

    def test_three_rows_create_a_continuous_material_band(self):
        edges = np.linspace(0.0, 1.0, 5)
        midpoint = 0.5 * (edges[:-1] + edges[1:])
        previous_edge = np.column_stack(
            (np.zeros(len(edges)), edges, np.zeros(len(edges)))
        )
        current_edge = previous_edge + [0.1, 0.0, 0.02]
        times = np.array([0.0, 0.05, 0.1])
        rows = np.array(
            [
                (1.0 + time) * (1.0 - midpoint**2)
                for time in times
            ]
        )
        result = newborn_halfwing_shedding_band(
            sheet_id="lev-row-1",
            vortex_family="LEV_SUCTION",
            previous_edge=previous_edge,
            current_edge=current_edge,
            span_edges=edges,
            time_nodes=times,
            strip_strength_rows=rows,
        )
        self.assertTrue(result.band.surface.continuity_report().compatible)
        self.assertTrue(all(item.passed for item in result.trace_reports))
        np.testing.assert_allclose(
            result.band.potential_jump_rows[:, 1::2],
            rows,
            atol=2.0e-14,
        )

    def test_midtime_row_and_shapes_are_mandatory(self):
        edges = np.linspace(0.0, 1.0, 5)
        geometry = np.column_stack(
            (np.zeros(len(edges)), edges, np.zeros(len(edges)))
        )
        with self.assertRaisesRegex(
            DistributedDoubletError,
            "strip_strength_rows",
        ):
            newborn_halfwing_shedding_band(
                sheet_id="missing-mid",
                vortex_family="TEV",
                previous_edge=geometry,
                current_edge=geometry + [0.1, 0.0, 0.0],
                span_edges=edges,
                time_nodes=[0.0, 0.5, 1.0],
                strip_strength_rows=np.zeros((2, 4)),
            )


if __name__ == "__main__":
    unittest.main()
