from pathlib import Path
import hashlib
import inspect
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import yaml

PLATFORM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLATFORM))

from claim_runtime import ClaimGraph, ClaimRuntimeError, RunConfig  # noqa: E402
from claim_runtime.p2_spatial_n3only_shadow import (  # noqa: E402
    P2SpatialN3OnlyShadow,
)


NODES = PLATFORM / "claim_nodes"
CLOSURE = "n3_spatial_edge_pressure_v1_shadow"
STAGE_ID = "N3.1j5"


def config(closure=CLOSURE):
    return RunConfig(closure=closure, values={}, sources={})


def zero_channels():
    return {
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
        "numerical_cycle_reduction": np.zeros(3),
    }


def internal_stage_spec(stage_id="STAGE", **overrides):
    source_path = inspect.getsourcefile(P2SpatialN3OnlyShadow)
    source_hash = "sha256:" + hashlib.sha256(
        Path(source_path).read_bytes()
    ).hexdigest()
    spec = {
        "id": stage_id,
        "state": "open",
        "freeze": False,
        "implementation": (
            "claim_runtime.p2_spatial_n3only_shadow:"
            "P2SpatialN3OnlyShadow"
        ),
        "implementation_version": "test",
        "implementation_hash": source_hash,
        "requires": [],
        "provides": [],
        "enabled_in": ["unit"],
        "runtime_role": "diagnostic_shadow",
        "runtime_binding": "internal_stage",
        "runtime_owner": "OWNER",
    }
    spec.update(overrides)
    return spec


def graph_from_internal_stages(stages):
    spec = {
        "id": "OWNER",
        "state": "partial",
        "freeze": False,
        "implementation": (
            "claim_runtime.components:TwistResponseObserver"
        ),
        "requires": ["solver_channels"],
        "provides": [],
        "enabled_in": ["unit"],
        "runtime_role": "diagnostic",
        "children": stages,
    }
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "owner.yaml"
        path.write_text(
            yaml.safe_dump(spec, sort_keys=False),
            encoding="utf-8",
        )
        return ClaimGraph.from_yaml(directory, config("unit"))


class InternalStageBindingTests(unittest.TestCase):
    def test_stage_is_traced_but_not_executed_or_booked(self):
        with mock.patch.object(
            P2SpatialN3OnlyShadow,
            "__init__",
            side_effect=AssertionError("internal stage must not be instantiated"),
        ):
            graph = graph_from_internal_stages([
                internal_stage_spec(stage_id=STAGE_ID)
            ])

        self.assertNotIn(STAGE_ID, graph.topology)
        self.assertEqual(graph.topology, ["OWNER"])

        context = graph.step(
            0,
            0.0,
            {"solver_channels": zero_channels()},
        )
        self.assertNotIn(STAGE_ID, context.ledger.by_node())
        self.assertNotIn(STAGE_ID, context.values)

        manifest = graph.manifest().to_dict()
        stages = manifest["internal_stages"]
        self.assertEqual(len(stages), 1)
        self.assertEqual(stages[0]["id"], STAGE_ID)
        self.assertEqual(stages[0]["runtime_binding"], "internal_stage")
        self.assertEqual(stages[0]["runtime_owner"], "OWNER")
        self.assertEqual(
            stages[0]["implementation_hash_scope"],
            "module_file",
        )
        self.assertRegex(
            stages[0]["implementation_hash"],
            r"^sha256:[0-9a-f]{64}$",
        )

    def test_empty_internal_stage_inventory_preserves_v41_manifest_shape(self):
        manifest = ClaimGraph.from_yaml(
            NODES,
            config("v41"),
        ).manifest().to_dict()
        self.assertNotIn("internal_stages", manifest)

    def test_pinned_internal_stage_hash_drift_is_fatal(self):
        with self.assertRaisesRegex(
            ClaimRuntimeError,
            "internal-stage implementation drift",
        ):
            graph_from_internal_stages([
                internal_stage_spec(
                    implementation_hash="sha256:" + "0" * 64,
                )
            ])

    def test_internal_stage_owner_must_be_enabled_executable_node(self):
        with self.assertRaisesRegex(
            ClaimRuntimeError,
            "owner 'MISSING' is not an enabled executable node",
        ):
            graph_from_internal_stages([
                internal_stage_spec(runtime_owner="MISSING")
            ])

    def test_internal_stage_exclusivity_is_enforced(self):
        with self.assertRaisesRegex(
            ClaimRuntimeError,
            "mutually exclusive nodes enabled",
        ):
            graph_from_internal_stages([
                internal_stage_spec(
                    "STAGE_A",
                    exclusive_with=["STAGE_B"],
                ),
                internal_stage_spec("STAGE_B"),
            ])

    def test_falsified_candidate_stage_cannot_rebind(self):
        with self.assertRaisesRegex(
            ClaimRuntimeError,
            "falsified internal stage cannot bind as candidate physics",
        ):
            ClaimGraph.from_yaml(NODES, config())
        with self.assertRaisesRegex(
            ClaimRuntimeError,
            "falsified internal stage cannot bind as candidate physics",
        ):
            graph_from_internal_stages([
                internal_stage_spec(state="falsified")
            ])


if __name__ == "__main__":
    unittest.main()
