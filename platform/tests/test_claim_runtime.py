from pathlib import Path
import sys
import unittest

import numpy as np

PLATFORM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLATFORM))

from claim_runtime import (  # noqa: E402
    ClaimComponent,
    ClaimGraph,
    ClaimRuntimeError,
    ForceContribution,
    ForceLedger,
    RunConfig,
)


NODES = PLATFORM / "claim_nodes"


def config(closure="v41"):
    return RunConfig(closure=closure, values={}, sources={})


class ClaimRuntimeTests(unittest.TestCase):
    def test_production_profiles_build_in_topological_order(self):
        v41 = ClaimGraph.from_yaml(NODES, config("v41"))
        self.assertLess(v41.topology.index("N1"), v41.topology.index("N2"))
        self.assertLess(v41.topology.index("N2"), v41.topology.index("N3"))
        self.assertLess(v41.topology.index("N6"), v41.topology.index("R0"))
        self.assertEqual(v41.topology[-1], "R0")
        self.assertNotIn("LEGACY", v41.topology)

        legacy = ClaimGraph.from_yaml(NODES, config("v4_legacy"))
        self.assertIn("LEGACY", legacy.topology)
        self.assertNotIn("N2", legacy.topology)
        self.assertNotIn("R0", legacy.topology)

    def test_force_ledger_rejects_duplicate_and_nonfinite(self):
        ledger = ForceLedger()
        ledger.add(ForceContribution("N1", "base", np.array([1.0, 0.0, 2.0])))
        with self.assertRaisesRegex(ClaimRuntimeError, "duplicate"):
            ledger.add(ForceContribution("N1", "base", np.zeros(3)))
        with self.assertRaisesRegex(ClaimRuntimeError, "non-finite"):
            ForceContribution("N2", "bad", np.array([np.nan, 0.0, 0.0]))

    def test_runtime_force_ledger_closes(self):
        graph = ClaimGraph.from_yaml(NODES, config("v41"))
        channels = {
            "uvlm": np.array([1.0, 0.0, 2.0]),
            "vortex_impulse": np.zeros(3),
            "leading_edge_suction": np.zeros(3),
            "uvlm_remainder": np.zeros(3),
            "separation": np.array([0.0, 0.0, -0.5]),
            "ds_vortex": np.array([0.1, 0.0, 0.4]),
            "vortex_normal": np.zeros(3),
            "ct_consistency": np.zeros(3),
            "rig_drag": np.array([0.5, 0.0, 0.0]),
            "viscous": np.zeros(3),
            "profile_drag": np.zeros(3),
            "numerical_cycle_reduction": np.array([0.2, 0.0, -0.1]),
        }
        context = graph.step(0, 0.0, {"solver_channels": channels})
        np.testing.assert_allclose(context.ledger.total_body(), [1.8, 0.0, 1.8])
        np.testing.assert_allclose(
            context.ledger.total_body({"physics", "necessary_physics"}),
            [1.6, 0.0, 1.9],
        )
        np.testing.assert_array_equal(
            context.values["r0_cycle_reduction"],
            context.ledger.by_node()["R0"][0]["body_force"],
        )
        self.assertEqual(context.ledger.by_node()["R0"][0]["role"], "diagnostic")
        self.assertEqual(
            context.ledger.by_node()["R0"][0]["channel"],
            "numerical_cycle_reduction",
        )

    def test_r0_requires_one_canonical_cycle_reduction_channel(self):
        graph = ClaimGraph.from_yaml(NODES, config("v41"))
        channels = {
            "uvlm": np.zeros(3),
            "vortex_impulse": np.zeros(3),
            "leading_edge_suction": np.zeros(3),
            "uvlm_remainder": np.zeros(3),
            "separation": np.zeros(3),
            "profile_drag": np.zeros(3),
            "ds_vortex": np.zeros(3),
            "vortex_normal": np.zeros(3),
            "ct_consistency": np.zeros(3),
            "rig_drag": np.zeros(3),
        }
        with self.assertRaisesRegex(
            ClaimRuntimeError, "missing canonical numerical_cycle_reduction"
        ):
            graph.step(0, 0.0, {"solver_channels": channels})

    def test_context_rejects_undeclared_output(self):
        class BadComponent(ClaimComponent):
            def step(self, context):
                from claim_runtime import NodeOutput
                return NodeOutput(values={"illegal": 1})

        component = BadComponent({"id": "BAD", "requires": [], "provides": []})
        from claim_runtime import StepContext
        context = StepContext()
        with self.assertRaisesRegex(ClaimRuntimeError, "undeclared"):
            context.publish("BAD", component.provides, component.step(context).values)


if __name__ == "__main__":
    unittest.main()
