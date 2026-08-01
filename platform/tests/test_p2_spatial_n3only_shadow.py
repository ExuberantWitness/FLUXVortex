import sys
import unittest
from pathlib import Path

import numpy as np

PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from claim_runtime.distributed_doublet import (  # noqa: E402
    DistributedDoubletError,
)
from claim_runtime.p2_spatial_n3only_shadow import (  # noqa: E402
    P2SpatialN3OnlyShadow,
    n3_only_unified_pressure_increment,
)
from claim_runtime.unified_panel_pressure import (  # noqa: E402
    structured_uvlm_surface_gradient,
    unified_panel_pressure,
)


class UnifiedN3OnlyDifferenceTests(unittest.TestCase):
    def setUp(self):
        self.density = 1.225
        self.dt = 0.02
        self.nc = 3
        self.ns = 2
        self.count = self.nc * self.ns
        self.velocity = np.column_stack(
            (
                np.linspace(7.5, 8.5, self.count),
                np.linspace(-0.2, 0.3, self.count),
                np.zeros(self.count),
            )
        )
        self.tc = np.tile([1.0, 0.0, 0.0], (self.count, 1))
        self.ts = np.tile([0.0, 1.0, 0.0], (self.count, 1))
        self.dx = np.full(self.count, 0.25)
        self.dy = np.full(self.count, 0.5)
        self.area = np.linspace(0.02, 0.04, self.count)
        self.normal = np.tile([0.0, 0.0, 1.0], (self.count, 1))
        self.gamma = np.array([0.4, 0.3, 0.25, 0.18, 0.1, 0.06])
        self.gamma_previous = 0.8 * self.gamma

    def evaluate(self, **updates):
        inputs = {
            "density": self.density,
            "dt": self.dt,
            "nc": self.nc,
            "ns": self.ns,
            "baseline_local_velocity": self.velocity,
            "p2_induced_velocity": np.zeros((self.count, 3)),
            "baseline_bound_gamma": self.gamma,
            "baseline_bound_previous": self.gamma_previous,
            "coupled_bound_gamma": self.gamma,
            "coupled_bound_previous": self.gamma_previous,
            "release_current": np.zeros(self.ns),
            "release_previous": np.zeros(self.ns),
            "panel_chord_tangent": self.tc,
            "panel_span_tangent": self.ts,
            "chord_step": self.dx,
            "span_step": self.dy,
            "area": self.area,
            "normal": self.normal,
        }
        inputs.update(updates)
        return n3_only_unified_pressure_increment(**inputs)

    def test_attached_limit_is_bitwise_zero(self):
        result = self.evaluate()
        np.testing.assert_array_equal(
            result.pressure_increment,
            np.zeros(self.count),
        )
        np.testing.assert_array_equal(
            result.force_increment,
            np.zeros((self.count, 3)),
        )
        self.assertTrue(result.guards.attached_zero_required)
        self.assertTrue(result.guards.attached_zero_bitwise)
        self.assertTrue(result.guards.passed)

    def test_bound_reaction_is_inside_the_same_pressure_difference(self):
        reaction = np.array([0.07, -0.02, 0.04, -0.01, 0.03, -0.015])
        coupled = self.gamma + reaction
        result = self.evaluate(coupled_bound_gamma=coupled)

        gradient = structured_uvlm_surface_gradient(
            coupled,
            chord_tangent=self.tc,
            span_tangent=self.ts,
            chord_step=self.dx,
            span_step=self.dy,
            nc=self.nc,
            ns=self.ns,
        )
        expected_coupled = unified_panel_pressure(
            density=self.density,
            local_velocity=self.velocity,
            surface_gradient=gradient,
            potential_rate_channels={
                "bound_unsteady": (
                    coupled - self.gamma_previous
                )
                / self.dt,
                "lev_release_unsteady": np.zeros(self.count),
            },
            area=self.area,
            normal=self.normal,
        )
        np.testing.assert_allclose(
            result.coupled_pressure,
            expected_coupled.total_pressure,
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_array_equal(
            result.bound_reaction,
            coupled - self.gamma,
        )
        self.assertGreater(
            np.max(np.abs(result.pressure_increment)),
            0.0,
        )
        self.assertTrue(result.guards.bound_reaction_included)
        self.assertTrue(result.guards.passed)

    def test_release_closing_row_is_not_clipped_or_gated_away(self):
        release_previous = np.array([0.05, -0.03])
        result = self.evaluate(
            release_current=np.zeros(self.ns),
            release_previous=release_previous,
        )
        expected_rate = -np.tile(release_previous, self.nc) / self.dt
        expected_pressure_delta = self.density * expected_rate
        np.testing.assert_allclose(
            result.pressure_increment,
            expected_pressure_delta,
            rtol=0.0,
            atol=2.0e-15,
        )
        self.assertGreater(
            np.max(np.abs(result.force_increment)),
            0.0,
        )
        self.assertFalse(result.guards.attached_zero_required)
        self.assertTrue(result.guards.passed)

    def test_outputs_are_read_only_and_nonfinite_input_fails(self):
        result = self.evaluate()
        self.assertFalse(result.pressure_increment.flags.writeable)
        self.assertFalse(result.force_increment.flags.writeable)
        with self.assertRaisesRegex(DistributedDoubletError, "finite"):
            self.evaluate(
                p2_induced_velocity=np.full(
                    (self.count, 3),
                    np.nan,
                )
            )


class StatefulN3OnlyShadowTests(unittest.TestCase):
    def setUp(self):
        self.nc = 2
        self.ns = 1
        self.count = self.nc * self.ns
        self.dt = 0.05
        self.shadow = P2SpatialN3OnlyShadow(
            nc=self.nc,
            ns=self.ns,
            span_edges=np.array([0.0, 1.0]),
            u_infinity=8.0,
            dt=self.dt,
            lesp_crit=0.2,
            quadrature_order=4,
            max_bands=4,
            mirror_halfwing=False,
        )
        self.aic = np.eye(self.count)
        self.gamma = np.array([0.1, 0.04])
        self.collocation = np.array(
            [[0.25, 0.5, 0.1], [0.75, 0.5, 0.1]]
        )
        self.normals = np.tile([0.0, 0.0, 1.0], (self.count, 1))
        self.leading_edge = np.array(
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )
        self.strip_tangent = np.array([[1.0, 0.0, 0.0]])
        self.suction_normal = np.array([[0.0, 0.0, 1.0]])
        self.tc = np.tile([1.0, 0.0, 0.0], (self.count, 1))
        self.ts = np.tile([0.0, 1.0, 0.0], (self.count, 1))
        self.dx = np.full(self.count, 0.5)
        self.dy = np.full(self.count, 1.0)
        self.area = np.full(self.count, 0.5)

    def advance(self, gamma=None, time=0.0):
        current = self.gamma if gamma is None else np.asarray(gamma)
        return self.shadow.advance_state(
            time=time,
            aic=self.aic,
            rhs_without_p2=current,
            baseline_bound_gamma=current,
            collocation=self.collocation,
            normals=self.normals,
            leading_edge=self.leading_edge,
            strip_chord_tangent=self.strip_tangent,
            suction_normal=self.suction_normal,
            alpha_rad=np.array([0.1]),
            chord=np.array([1.0]),
            delta_x_front=np.array([0.5]),
        )

    def pressure(self, baseline_previous):
        return self.shadow.pressure_increment(
            density=1.225,
            baseline_local_velocity=np.tile(
                [8.0, 0.0, 0.0],
                (self.count, 1),
            ),
            baseline_bound_previous=baseline_previous,
            panel_chord_tangent=self.tc,
            panel_span_tangent=self.ts,
            chord_step=self.dx,
            span_step=self.dy,
            area=self.area,
            normal=self.normals,
        )

    def test_no_release_state_is_exactly_zero_and_history_advances(self):
        aic_before = self.aic.copy()
        gamma_before = self.gamma.copy()
        first = self.advance()
        self.assertTrue(first.inventory_absent)
        first_pressure = self.pressure(np.zeros(self.count))
        np.testing.assert_array_equal(
            first_pressure.force_increment,
            np.zeros((self.count, 3)),
        )

        second_gamma = np.array([0.08, 0.03])
        second = self.advance(second_gamma, time=self.dt)
        self.assertTrue(second.inventory_absent)
        second_pressure = self.pressure(self.gamma)
        np.testing.assert_array_equal(
            second_pressure.pressure_increment,
            np.zeros(self.count),
        )
        np.testing.assert_array_equal(self.aic, aic_before)
        np.testing.assert_array_equal(self.gamma, gamma_before)
        diagnostics = self.shadow.diagnostics()
        self.assertEqual(diagnostics["advanced_steps"], 2)
        self.assertEqual(diagnostics["pressure_steps"], 2)
        self.assertEqual(
            diagnostics["private_bound_previous"],
            second_gamma.tolist(),
        )
        self.assertFalse(diagnostics["self_advection_included"])
        self.assertFalse(diagnostics["pressure_clipping_included"])
        self.assertFalse(diagnostics["target_data_access"])

    def test_ordering_guards_prevent_state_overwrite(self):
        self.advance()
        with self.assertRaisesRegex(
            DistributedDoubletError,
            "consume",
        ):
            self.advance()
        with self.assertRaisesRegex(
            DistributedDoubletError,
            "precede convect",
        ):
            self.shadow.convect(
                lambda points: np.zeros_like(points)
            )
        self.pressure(np.zeros(self.count))
        self.shadow.convect(lambda points: np.zeros_like(points))
        self.assertEqual(
            self.shadow.diagnostics()["convection_steps"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
