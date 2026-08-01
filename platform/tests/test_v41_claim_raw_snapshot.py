import sys
import unittest
from pathlib import Path

import numpy as np

PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))


class V41ClaimRawSnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import warp as wp

        wp.init()
        try:
            wp.get_device("cuda:0")
        except Exception as exc:  # pragma: no cover - CPU-only CI
            raise unittest.SkipTest(f"FLUXV production solver needs CUDA: {exc}")

        from _v2_robo import gpu_run_twist
        from lb_sweep118 import BASE

        cls.steps_per_cycle = 12
        config = dict(BASE)
        config.update(
            U=8.0,
            aoa_deg=5.0,
            freq=2.6,
            twist_amp_deg=7.5,
            nc=2,
            ns=4,
            n_cycle=1,
            steps_per_cycle=cls.steps_per_cycle,
            wake_rows=cls.steps_per_cycle,
            closure="v41",
        )
        cls.without_observer = gpu_run_twist(**config)
        cls.raw = []
        cls.with_observer = gpu_run_twist(
            **config, claim_raw_out=cls.raw
        )

    def test_observer_is_force_isolated(self):
        numeric_fields = (
            "L_wind",
            "T_wind",
            "Fx_body",
            "Fz_body",
            "L_inst",
            "T_inst",
            "Lh_bern",
            "Xh_bern",
            "Lh_stall",
            "Xh_stall",
            "Lh_ds",
            "Xh_ds",
        )
        for name in numeric_fields:
            np.testing.assert_array_equal(
                np.asarray(self.with_observer[name]),
                np.asarray(self.without_observer[name]),
                err_msg=name,
            )

    def test_last_cycle_shape_and_finite_values(self):
        self.assertEqual(len(self.raw), self.steps_per_cycle)
        for expected_step, record in enumerate(self.raw):
            self.assertEqual(record["last_cycle_step"], expected_step)
            self.assertEqual(
                record["cycle_index"],
                record["step"] // self.steps_per_cycle,
            )
            self.assertEqual(record["snapshot_phase"], "post_force_pre_shed")
            np.testing.assert_allclose(
                record["y_ref_center_m"],
                0.5 * (
                    record["y_ref_edge_m"][:-1]
                    + record["y_ref_edge_m"][1:]
                ),
                rtol=0.0,
                atol=1.0e-15,
            )
            self.assertEqual(record["n1"]["panel_force_body_N"].shape, (8, 3))
            self.assertEqual(
                record["n3"]["ds_panel_force_solver_legacy_N"].shape,
                (8, 3),
            )
            for group_name in ("n1", "n2", "n3"):
                for name, value in record[group_name].items():
                    array = np.asarray(value)
                    if array.dtype.kind in "biufc":
                        self.assertTrue(
                            np.isfinite(array).all(),
                            msg=f"{group_name}.{name}",
                        )
            self.assertNotIn("wr", record)
            self.assertNotIn("wg", record)
            self.assertNotIn("particles", record)

    def test_node_and_reported_force_ledgers_close(self):
        for record in self.raw:
            n1 = record["n1"]
            n2 = record["n2"]
            n3 = record["n3"]
            n1_panel_resultant = np.sum(
                n1["panel_force_body_N"], axis=0
            )
            # The production accumulator books x/z only; the one-wing
            # spanwise component is cancelled by the mirrored wing.
            np.testing.assert_allclose(
                n1_panel_resultant[[0, 2]],
                n1["bernoulli_booked_solver_accumulator_N"][[0, 2]],
                rtol=0.0,
                atol=2.0e-14,
            )
            n3_panel_resultant = np.sum(
                n3["ds_panel_force_solver_legacy_N"], axis=0
            )
            np.testing.assert_allclose(
                n3_panel_resultant[[0, 2]],
                n3["ds_booked_solver_accumulator_N"][[0, 2]],
                rtol=0.0,
                atol=2.0e-14,
            )
            reconstructed = (
                n1["booked_solver_accumulator_total_N"]
                + n2["booked_solver_accumulator_total_N"]
                + n3["booked_solver_accumulator_total_N"]
            )
            np.testing.assert_allclose(
                reconstructed,
                record["total_solver_accumulator_body_force_N"],
                rtol=0.0,
                atol=2.0e-14,
            )

        np.testing.assert_array_equal(
            np.asarray(
                [record["reported_pair_wind_lift_N"] for record in self.raw]
            ),
            np.asarray(self.with_observer["L_inst"])[-self.steps_per_cycle:],
        )
        np.testing.assert_array_equal(
            np.asarray(
                [record["reported_pair_wind_thrust_N"] for record in self.raw]
            ),
            np.asarray(self.with_observer["T_inst"])[-self.steps_per_cycle:],
        )

    def test_replay_validity_contract_accepts_observer_snapshot(self):
        from d3_claim_replay import _validate_raw

        report = _validate_raw(
            self.raw,
            self.raw,
            self.with_observer,
            self.steps_per_cycle,
            1.0 / (2.6 * self.steps_per_cycle),
        )
        self.assertTrue(report["passed"], msg=report)
        self.assertLessEqual(
            report["n3_force_scope_relation_error_N"], 2.0e-12
        )


if __name__ == "__main__":
    unittest.main()
