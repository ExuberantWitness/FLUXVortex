"""E0 exit gate: production-RHS observer on/off parity (FINAL_PROPOSAL §5.1).

Builds the SAME formal A16 case twice from identical construction, runs a
short slice with the RHS observer DISABLED and ENABLED, and compares the
committed trajectories bit-exactly.  The observer is contractually
read-only; any bit difference fails the gate and forbids every downstream
load conclusion (§6 result-to-decision row 1).

Also validates the tape itself: per formal substep the identity flag,
component norms, algorithmic work and predictor-lag diagnostics exist and
are finite.
"""

from __future__ import annotations

import math
import unittest

import torch
import warp as wp

torch.cuda.init()
if not torch.cuda.is_available():
    raise unittest.SkipTest("E0 parity requires CUDA")

from fluxvortex.cases.rojratsirikul2011 import ROJ_A16_PRIMARY
from fluxvortex.runtime.case_runner import RojratsirikulCaseRunner

STEPS = 2


def _run(observer_enabled: bool):
    runner = RojratsirikulCaseRunner(
        ROJ_A16_PRIMARY, reference_tangent_refresh_rtol=0.0
    )
    runner.build()
    runner.rhs_observer_enabled = observer_enabled
    payload = runner.run(max_aero_steps=STEPS, output=None)
    state = {
        "q": wp.to_torch(runner.owner.state).clone(),
        "v": wp.to_torch(runner.owner.velocity).clone(),
        "a": wp.to_torch(runner.owner.acceleration).clone(),
        "gamma": runner.owner.aerodynamic.state.gamma_bound.clone(),
        "wake_gamma": runner.owner.aerodynamic.state.wake_gamma.clone(),
        "payload": payload,
        "tape": list(runner._rhs_tape),
    }
    return state


class TestE0RhsObserverParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.off = _run(False)
        cls.on = _run(True)

    def test_trajectories_bit_identical(self):
        for key in ("q", "v", "a", "gamma", "wake_gamma"):
            self.assertTrue(
                torch.equal(self.off[key], self.on[key]),
                f"observer changed committed {key}",
            )

    def test_formal_outputs_identical(self):
        for key in ("mean_zmax_over_c", "mean_Cn"):
            self.assertEqual(self.off["payload"][key], self.on["payload"][key])

    def test_tape_contract(self):
        tape = self.on["tape"]
        self.assertGreater(len(tape), 0)
        formal = [e for e in tape if e["formal_replay"]]
        # 2 steps x 10 substeps of formal replay at minimum
        self.assertGreaterEqual(len(formal), 2 * 10)
        for entry in formal:
            self.assertEqual(entry["identity"], "formal_committed")
            for key in (
                "constant_norm",
                "velocity_norm",
                "mf1_action_norm",
                "total_aero_norm",
                "dq_norm",
                "w_algorithmic",
                "dw_predictor_lag",
            ):
                self.assertIn(key, entry)
                self.assertTrue(math.isfinite(entry[key]), f"{key} not finite")

    def test_payload_evidence_present(self):
        evidence = self.on["payload"]["rhs_observer_evidence"]
        self.assertTrue(evidence["enabled"])
        self.assertGreater(evidence.get("formal_substep_entries", 0), 0)
        self.assertIn("w_algorithmic", evidence)
        self.assertIn("dw_predictor_lag", evidence)


if __name__ == "__main__":
    unittest.main()
