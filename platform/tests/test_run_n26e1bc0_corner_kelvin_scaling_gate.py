import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


PLATFORM = Path(__file__).resolve().parents[1]
RUNNER = PLATFORM / "run_n26e1bc0_corner_kelvin_scaling_gate.py"
SPEC = importlib.util.spec_from_file_location(
    "run_n26e1bc0_corner_kelvin_scaling_gate",
    RUNNER,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class CornerKelvinScalingGateTests(unittest.TestCase):
    def test_frozen_axes_and_common_anchor_are_exact(self):
        spatial = MODULE.spatial_cases()
        temporal = MODULE.temporal_cases()
        self.assertEqual(
            [(case.panels_per_side, case.time_step_s) for case in spatial],
            [(64, 0.00625), (128, 0.00625), (256, 0.00625)],
        )
        self.assertEqual(
            [(case.panels_per_side, case.time_step_s) for case in temporal],
            [
                (256, 0.025),
                (256, 0.0125),
                (256, 0.00625),
                (256, 0.003125),
            ],
        )
        self.assertEqual(
            [case.ramp_steps for case in temporal],
            [16, 32, 64, 128],
        )
        self.assertEqual(
            [case.observation_step for case in temporal],
            [8, 16, 32, 64],
        )
        self.assertEqual(len(MODULE.frozen_cases()), 6)
        self.assertEqual(MODULE.COMMON_ANCHOR, (256, 0.003125))
        with self.assertRaises(ValueError):
            MODULE.ScalingCase(128, 0.0125)
        with self.assertRaises(ValueError):
            MODULE.ScalingCase(256, 0.01)

    def test_preregistration_hash_was_frozen_before_implementation(self):
        self.assertEqual(
            MODULE._sha256(MODULE.PREREGISTRATION),
            MODULE.EXPECTED_PREREGISTRATION_SHA256,
        )
        manifest = MODULE._protocol_manifest()
        self.assertEqual(
            manifest["axes"]["common_anchor_case_id"],
            MODULE.ScalingCase(256, 0.003125).case_id,
        )
        self.assertEqual(
            manifest["observables"]["algebraic_residuals"],
            list(MODULE.ALGEBRAIC_KEYS),
        )

    def test_terminal_radius_and_stage_extractor_use_only_allowed_fields(self):
        surface = SimpleNamespace(
            upper_nodes=np.array(((0.0, 0.0), (1.0, 0.0))),
            panel_midpoints=np.array(
                ((0.75, -0.1), (0.75, 0.1)),
            ),
        )
        radius = MODULE._terminal_midpoint_radius(surface)
        self.assertAlmostEqual(
            radius["canonical_m"],
            math.sqrt(0.25**2 + 0.1**2),
        )
        solution = SimpleNamespace(
            common_te_diagnostic=SimpleNamespace(
                lower_downstream_trace=4.0,
                upper_downstream_trace=2.0,
                mean_downstream_trace=3.0,
            ),
            bound_circulation_ccw=-1.25,
            newborn_circulation_ccw=0.25,
            near_wake_segment=SimpleNamespace(
                orientation_side="lower",
            ),
        )
        result = MODULE._stage_observables(
            solution,
            previous_bound_circulation_ccw=-1.0,
            terminal_radius=radius,
        )
        self.assertEqual(
            result["actual_newborn_circulation_ccw_m2_s"],
            0.25,
        )
        self.assertEqual(result["birth_identity_residual_m2_s"], 0.0)
        self.assertEqual(result["branch_identity"], "lower")
        self.assertAlmostEqual(
            result["modal_A_u_over_r_beta"]["mean"],
            3.0
            / radius["canonical_m"]
            ** MODULE.REGULAR_VELOCITY_EXPONENT,
        )

    def test_four_point_log_ols_uses_all_levels_and_reports_local_orders(self):
        exponent = 1.125
        time_steps = MODULE.TEMPORAL_TIME_STEPS
        births = [2.5 * time_step**exponent for time_step in time_steps]
        fit = MODULE.four_point_log_ols(time_steps, births)
        self.assertAlmostEqual(fit["slope_p_K"], exponent, places=13)
        self.assertEqual(fit["design_rank"], 2)
        self.assertEqual(len(fit["local_orders_consecutive_pairs"]), 3)
        for order in fit["local_orders_consecutive_pairs"]:
            self.assertAlmostEqual(order, exponent, places=13)
        with self.assertRaises(ValueError):
            MODULE.four_point_log_ols(time_steps[:3], births[:3])
        with self.assertRaises(ValueError):
            MODULE.four_point_log_ols(time_steps, [*births[:3], 0.0])

    @staticmethod
    def _synthetic_case(case, *, exponent):
        radius_by_panel = {
            64: 4.0e-4,
            128: 1.0e-4,
            256: 2.5e-5,
        }
        radius = radius_by_panel[case.panels_per_side]
        modal = {
            "lower": 10.0,
            "upper": 9.5,
            "mean": 9.75,
        }
        birth = 3.0 * case.time_step_s**exponent
        residuals = {
            key: 1.0e-12 for key in MODULE.ALGEBRAIC_KEYS
        }
        return {
            **case.as_dict(),
            "status": "completed",
            "history_time_alignment_error_s": 0.0,
            "observables_at_t_star": {
                "terminal_midpoint_radius": {
                    "lower_m": radius,
                    "upper_m": radius,
                    "canonical_m": radius,
                    "symmetry_error_m": 0.0,
                },
                "terminal_traces_m_s": {
                    name: value
                    * radius ** MODULE.REGULAR_VELOCITY_EXPONENT
                    for name, value in modal.items()
                },
                "modal_A_u_over_r_beta": modal,
                "actual_newborn_circulation_ccw_m2_s": birth,
                "solver_newborn_circulation_ccw_m2_s": birth,
                "birth_identity_residual_m2_s": 0.0,
                "birth_magnitude_over_Uc": abs(birth) / 9.0,
                "branch_identity": "lower",
            },
            "maximum_algebraic_residuals": residuals,
            "branch_audit": {
                "unique_identities": ["lower"],
                "internally_consistent": True,
                "final_identity": "lower",
            },
            "finite_health": True,
            "algebraic_gate_pass": True,
            "generic_birth_gate_pass": True,
        }

    def test_evaluator_separates_protocol_failure_from_physics(self):
        failed = MODULE.evaluate_gate(
            {},
            preregistration_hash_matches=False,
        )
        self.assertEqual(failed["verdict"], "PROTOCOL-NO-GO")
        self.assertFalse(failed["physics"]["available"])
        self.assertEqual(failed["physics"]["status"], "open")
        self.assertEqual(failed["claim_state"], "open")

    def test_evaluator_can_reach_representation_go_only_after_health(self):
        exponent = MODULE.REGULAR_ONLY_BIRTH_EXPONENT
        cases = {
            case.case_id: self._synthetic_case(
                case,
                exponent=exponent,
            )
            for case in MODULE.frozen_cases()
        }
        decision = MODULE.evaluate_gate(
            cases,
            preregistration_hash_matches=True,
        )
        self.assertTrue(decision["protocol"]["pass"])
        self.assertEqual(decision["verdict"], "REPRESENTATION-GO")
        self.assertEqual(decision["claim_state"], "parent_open")
        self.assertTrue(
            decision["physics"]["necessary_scaling_gate_pass"]
        )

    def test_generic_kelvin_order_one_is_physics_no_go_not_protocol_no_go(self):
        cases = {
            case.case_id: self._synthetic_case(case, exponent=1.0)
            for case in MODULE.frozen_cases()
        }
        decision = MODULE.evaluate_gate(
            cases,
            preregistration_hash_matches=True,
        )
        self.assertTrue(decision["protocol"]["pass"])
        self.assertEqual(decision["verdict"], "PHYSICS-NO-GO")
        self.assertEqual(
            decision["claim_state"],
            "N2.6e1bc0_regular_corner_only_falsified_frozen",
        )
        self.assertAlmostEqual(
            decision["physics"]["time_scaling"]["slope_p_K"],
            1.0,
            places=13,
        )

    def test_execute_gate_fails_closed_before_cases_if_prereg_hash_drifts(self):
        with patch.object(MODULE, "_sha256", return_value="drift"):
            result = MODULE.execute_gate()
        self.assertEqual(result["cases"], {})
        self.assertEqual(
            result["decision"]["verdict"],
            "PROTOCOL-NO-GO",
        )

    def test_result_writer_rejects_nonfinite_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(ValueError):
                MODULE.write_results(
                    {"bad": math.nan},
                    json_path=root / "result.json",
                    markdown_path=root / "result.md",
                )


if __name__ == "__main__":
    unittest.main()
