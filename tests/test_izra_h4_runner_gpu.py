"""H4: focused Figure 14 mandatory runner — GT, scoring, LESP, run, CT sign.

Covers tests/test_izra_h4_runner_gpu.py scope from the H4 task:
- load_gt_markers(): exactly 14 Scherer markers, correct fields/replicates;
- score_all(): synthetic predictions reproduce MAE/RMSE/bias by hand, with
  each replicate scored independently;
- compute_lesp_crit(0.0607, 310000) ~= 0.2393 (the frozen Scherer threshold);
- run_one_condition() on IZRA-15-045 with a small grid (4x12, 32 steps per
  cycle, 1 cycle): runs, finite CT, physics evidence intact;
- the CT sign convention: freestream is +x in the native frame, so the
  propulsive direction is -x — mean world Fx < 0 and CT_raw = -<Fx>/qS > 0
  on a real thrust-producing condition.

CUDA float64 is required for the run tests; the GT/scoring/LESP tests are
pure CPU contract gates.
"""
from __future__ import annotations

import math
import os
import unittest

import torch
import warp as wp

from fluxvortex.cases.izraelevitz2017 import IZRA_CASES, LESP_THRESHOLD
from fluxvortex.warp_fsi.q16_flux_v5m_native import compute_lesp_crit

import reproduce_izraelevitz2017_fig14_v5m_mandatory as h4

DEVICE = os.environ.get("FLUXV_DEVICE", "cuda:0")
SMALL_GRID = dict(chordwise_panels=4, spanwise_panels=12, steps_per_cycle=32)
CUDA_OK = torch.cuda.is_available() and wp.is_cuda_available()


class LoadGtMarkersTest(unittest.TestCase):
    """The 14-marker Scherer ground truth loads with the frozen structure."""

    @classmethod
    def setUpClass(cls):
        cls.markers = h4.load_gt_markers()

    def test_exactly_14_markers_and_12_conditions(self):
        self.assertEqual(len(self.markers), 14)
        conditions = {(m["theta_max_deg"], m["phase_offset_deg"]) for m in self.markers}
        self.assertEqual(len(conditions), 12)
        self.assertEqual(
            sorted(conditions),
            [
                (15.0, 15.0), (15.0, 30.0), (15.0, 45.0), (15.0, 60.0),
                (15.0, 75.0), (15.0, 90.0), (15.0, 105.0),
                (25.0, 45.0), (25.0, 60.0), (25.0, 75.0),
                (25.0, 90.0), (25.0, 105.0),
            ],
        )

    def test_correct_fields(self):
        expected = {
            "case_id",
            "theta_max_deg",
            "phase_offset_deg",
            "replicate",
            "ct",
            "ct_error_minus",
            "ct_error_plus",
        }
        for marker in self.markers:
            self.assertEqual(set(marker), expected)
            self.assertIsInstance(marker["replicate"], int)
            for name in ("theta_max_deg", "phase_offset_deg", "ct", "ct_error_minus", "ct_error_plus"):
                self.assertIsInstance(marker[name], float)
                self.assertTrue(math.isfinite(marker[name]))
            self.assertEqual(
                marker["case_id"],
                f"IZRA-{int(marker['theta_max_deg']):02d}-{int(marker['phase_offset_deg']):03d}",
            )

    def test_duplicate_replicates_kept_not_averaged(self):
        counts = {}
        for marker in self.markers:
            key = (marker["theta_max_deg"], marker["phase_offset_deg"])
            counts[key] = counts.get(key, 0) + 1
        self.assertEqual(counts[(15.0, 15.0)], 2)
        self.assertEqual(counts[(15.0, 75.0)], 2)
        self.assertEqual(sum(counts.values()), 14)

    def test_frozen_values(self):
        by_key = {(m["theta_max_deg"], m["phase_offset_deg"], m["replicate"]): m for m in self.markers}
        self.assertAlmostEqual(by_key[(15.0, 15.0, 1)]["ct"], 0.123091624, places=9)
        self.assertAlmostEqual(by_key[(15.0, 15.0, 2)]["ct"], 0.144850206, places=9)
        self.assertAlmostEqual(by_key[(15.0, 75.0, 2)]["ct"], 0.205152638, places=9)
        self.assertAlmostEqual(by_key[(25.0, 105.0, 1)]["ct"], 0.012053646, places=9)
        self.assertAlmostEqual(by_key[(15.0, 15.0, 1)]["ct_error_minus"], 0.011190235, places=9)


class ScoreAllTest(unittest.TestCase):
    """Synthetic predictions reproduce MAE/RMSE/bias with replicate weighting."""

    @staticmethod
    def _markers():
        # 5 markers over 4 conditions; (15,15) carries two replicates that a
        # single prediction must score independently.
        return [
            {"theta_max_deg": 15.0, "phase_offset_deg": 15.0, "replicate": 1, "ct": 0.10, "ct_error_minus": 0.01, "ct_error_plus": 0.01, "case_id": "IZRA-15-015"},
            {"theta_max_deg": 15.0, "phase_offset_deg": 15.0, "replicate": 2, "ct": 0.12, "ct_error_minus": 0.01, "ct_error_plus": 0.01, "case_id": "IZRA-15-015"},
            {"theta_max_deg": 15.0, "phase_offset_deg": 30.0, "replicate": 1, "ct": 0.20, "ct_error_minus": 0.01, "ct_error_plus": 0.01, "case_id": "IZRA-15-030"},
            {"theta_max_deg": 15.0, "phase_offset_deg": 45.0, "replicate": 1, "ct": 0.30, "ct_error_minus": 0.01, "ct_error_plus": 0.01, "case_id": "IZRA-15-045"},
            {"theta_max_deg": 25.0, "phase_offset_deg": 60.0, "replicate": 1, "ct": 0.40, "ct_error_minus": 0.01, "ct_error_plus": 0.01, "case_id": "IZRA-25-060"},
        ]

    def test_hand_computed_metrics(self):
        predictions = {
            (15.0, 15.0): 0.11,
            (15.0, 30.0): 0.23,
            (15.0, 45.0): 0.28,
            (25.0, 60.0): 0.44,
        }
        # errors: +0.01, -0.01, +0.03, -0.02, +0.04 (the (15,15) prediction
        # pairs with BOTH replicates — never pre-averaged).
        metrics = h4.score_all(predictions, self._markers())
        self.assertEqual(metrics["n_markers"], 5)
        self.assertAlmostEqual(metrics["mae_ct"], 0.022, places=12)
        self.assertAlmostEqual(metrics["rmse_ct"], math.sqrt(6.2e-4), places=12)
        self.assertAlmostEqual(metrics["bias_ct"], 0.01, places=12)
        self.assertAlmostEqual(metrics["max_abs_error_ct"], 0.04, places=12)
        self.assertEqual(metrics["max_abs_error_condition"], [25.0, 60.0])
        self.assertEqual(len(metrics["per_marker"]), 5)

    def test_perfect_prediction_zero_error(self):
        # Use only unique conditions here: a duplicated condition cannot be
        # matched perfectly by ONE prediction (its replicates differ), which
        # is precisely why each replicate is scored independently above.
        markers = [m for m in self._markers() if m["replicate"] == 1]
        predictions = {(m["theta_max_deg"], m["phase_offset_deg"]): m["ct"] for m in markers}
        metrics = h4.score_all(predictions, markers)
        self.assertEqual(metrics["n_markers"], 4)
        self.assertAlmostEqual(metrics["mae_ct"], 0.0, places=15)
        self.assertAlmostEqual(metrics["rmse_ct"], 0.0, places=15)
        self.assertAlmostEqual(metrics["bias_ct"], 0.0, places=15)
        self.assertTrue(all(row["within_error_bar"] for row in metrics["per_marker"]))

    def test_missing_prediction_fails_closed(self):
        with self.assertRaises(KeyError):
            h4.score_all({(15.0, 15.0): 0.1}, self._markers())


class LespThresholdTest(unittest.TestCase):
    """The physics-based LESPcrit reproduces the frozen Scherer 0.2393."""

    def test_compute_lesp_crit_hits_frozen_threshold(self):
        value = compute_lesp_crit(h4.LESP_THICKNESS_RATIO, h4.LESP_REYNOLDS)
        self.assertAlmostEqual(value, 0.2393, places=4)
        self.assertAlmostEqual(value, LESP_THRESHOLD, delta=5.0e-5)

    def test_runner_constants_frozen(self):
        self.assertEqual(h4.LESP_THICKNESS_RATIO, 0.0607)
        self.assertEqual(h4.LESP_REYNOLDS, 310000.0)
        self.assertEqual(h4.THRUST_SIGN, -1.0)
        self.assertEqual(h4.FROZEN_MAE_GATE, 0.01745211311116545)

    def test_invalid_inputs_raise(self):
        with self.assertRaises(ValueError):
            compute_lesp_crit(-0.1, 310000.0)
        with self.assertRaises(ValueError):
            compute_lesp_crit(0.06, 0.0)


@unittest.skipUnless(CUDA_OK, "CUDA required")
class RunOneConditionSmokeTest(unittest.TestCase):
    """IZRA-15-045, 4x12 grid, 32 steps/cycle, 1 cycle — runs, finite CT."""

    @classmethod
    def setUpClass(cls):
        cls.result = h4.run_one_condition(
            "IZRA-15-045", device=DEVICE, cycles=1, **SMALL_GRID
        )

    def test_run_completes_with_finite_ct(self):
        result = self.result
        self.assertEqual(result["case_id"], "IZRA-15-045")
        self.assertEqual(result["chordwise_panels"], 4)
        self.assertEqual(result["spanwise_panels"], 12)
        self.assertEqual(result["steps_per_cycle"], 32)
        self.assertEqual(result["steps_total"], 32)
        self.assertEqual(result["scored_sample_count"], 32)
        self.assertEqual(len(result["force_history"]), 32)
        for fx, fy, fz in result["force_history"]:
            self.assertTrue(math.isfinite(fx) and math.isfinite(fy) and math.isfinite(fz))
        self.assertTrue(math.isfinite(result["ct_raw"]))
        self.assertTrue(math.isfinite(result["ct"]))
        self.assertAlmostEqual(
            result["ct"], result["ct_raw"] - IZRA_CASES["IZRA-15-045"].profile_drag_coefficient, places=15
        )
        self.assertGreater(result["elapsed_seconds"], 0.0)

    def test_physics_evidence_recorded(self):
        physics = self.result["physics"]
        self.assertIs(physics["enable_lev"], True)
        self.assertIs(physics["joint_tev"], True)
        self.assertIs(physics["prescribed_wake"], False)
        self.assertAlmostEqual(physics["effective_lesp_crit"], 0.2393, places=4)
        self.assertEqual(physics["proposal_count"], 32)
        self.assertEqual(physics["accepted_commit_count"], 32)
        self.assertEqual(physics["tev_shed_count"], 32)
        self.assertEqual(physics["free_wake_convection_count"], 31)
        self.assertTrue(physics["parent_state_unchanged_on_proposal"])
        self.assertIs(physics["posthoc_separation_delta_applied"], False)
        self.assertTrue(physics["cuda_float64_all_steps"])
        self.assertIn("RigidAuthorLoadAssembler", physics["surface_load_owner"])


@unittest.skipUnless(CUDA_OK, "CUDA required")
class ThrustSignConventionTest(unittest.TestCase):
    """Coordinate audit on a real condition (HANDOFF §4): thrust direction.

    The native frame carries the freestream along +x, so propulsion pushes
    the wing upstream (-x): THRUST_SIGN = -1.  Audited on IZRA-15-090 (the
    strongest clean thrust phasing in the Scherer set, CT_exp ~= 0.19):
    (a) the last-cycle mean world Fx must be negative and CT_raw = -<Fx>/qS
    positive; (b) at the maximum-heave-rate events of the last cycle the
    chordwise force must stay negative while the wing carries a substantial
    load (|Fz|/qS >> 0) — the instantaneous signature of thrust production
    (F_x = rho*Gamma*zdot < 0 in BOTH half-cycles), which no sign flip or
    absolute value could fake.
    """

    @classmethod
    def setUpClass(cls):
        # 3 cycles so the scored (last) cycle is past the impulsive start.
        cls.case = IZRA_CASES["IZRA-15-090"]
        cls.result = h4.run_one_condition(
            "IZRA-15-090", device=DEVICE, cycles=3, **SMALL_GRID
        )

    def _last_cycle_rows(self):
        case, result = self.case, self.result
        spc = result["steps_per_cycle"]
        cycle_start_step = (result["cycles"] - 1) * spc
        rows = []
        for k, (fx, fy, fz) in enumerate(result["force_history"][-spc:], start=1):
            t = (cycle_start_step + k) * result["aerodynamic_dt_s"]
            zdot = -case.heave_amplitude_m * case.omega_rad_s * math.sin(
                case.omega_rad_s * t
            )
            rows.append((zdot, fx, fz))
        return rows

    def test_mean_world_fx_is_negative_upstream_thrust(self):
        result = self.result
        self.assertEqual(result["thrust_sign"], -1.0)
        self.assertLess(result["mean_force_x_n"], 0.0)

    def test_ct_raw_is_positive_thrust_in_plausible_range(self):
        result = self.result
        self.assertGreater(result["ct_raw"], 0.0)
        # Coarse-grid sanity: the same order as the Scherer thrust
        # coefficient (experiment 0.191 including the -0.057 Cd0 shift).
        self.assertLess(result["ct_raw"], 1.0)

    def test_max_heave_rate_events_are_thrust_signed(self):
        rows = sorted(self._last_cycle_rows(), key=lambda row: -abs(row[0]))
        for zdot, fx, fz in rows[:6]:
            self.assertLess(fx, 0.0, f"thrust event lost upstream sign (zdot={zdot:+.3f})")
            self.assertGreater(abs(fz) / self.result["q_area_n"], 0.5)

    def test_q_area_matches_full_wing_dynamic_pressure(self):
        case = self.case
        expected = 0.5 * case.rho_kg_m3 * case.freestream_m_s**2 * case.area_m2
        self.assertAlmostEqual(self.result["q_area_n"], expected, places=9)


if __name__ == "__main__":
    unittest.main()
