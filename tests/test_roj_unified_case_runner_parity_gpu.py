"""P2 migration parity: legacy direct loop vs unified CaseRunner (handoff §9).

Both paths drive the SAME frozen numerics (Q16NativeV5MFSIStepper over
Q16CudaNewmarkStepper + Q16NativeV5MSolver) on the formal grid for the first
8 aerodynamic steps of ROJ11-A16.  The unified CaseRunner adds only wrappers
(WorldOwner/GlobalTransaction/V5M3DStepper verification/frame evaluation) —
so every physical observable must be BIT-IDENTICAL.  Any difference here
means the migration changed physics.
"""

from __future__ import annotations

import math
import os
import unittest

import numpy as np
import torch
import warp as wp

from fluxvortex.cases.rojratsirikul2011 import ROJ_A16_PRIMARY
from fluxvortex.runtime.case_runner import (
    RojratsirikulCaseRunner,
    apply_freestream,
)
from fluxvortex.warp_fsi import config
from fluxvortex.warp_fsi.q16_flux_v5m_native import (
    NativeV5MConfig,
    Q16NativeV5MSolver,
    Q16NativeV5MSurface,
)
from fluxvortex.warp_fsi.q16_flux_v5m_native_fsi import (
    Q16NativeV5MFSIOwner,
    Q16NativeV5MFSIStepper,
)
from fluxvortex.warp_fsi.q16_structural_solver import Q16CudaNewmarkStepper
from forward_flight_benchmarks.rojratsirikul2011_q16 import (
    FORMAL_AERO_GRID,
    FORMAL_Q16_GRID,
    make_rojratsirikul2011_q16_model,
    normal_force_coefficient,
    plate_normal,
)

PARITY_STEPS = 8


def _legacy_stack():
    """The retired inline production construction, verbatim (handoff §8.2)."""
    from forward_flight_benchmarks.rojratsirikul2011_q16 import (
        ROJRATSIRIKUL2011_CASES,
    )

    case = ROJRATSIRIKUL2011_CASES["ROJ11-A16"]
    mesh, model, boundary, _ = make_rojratsirikul2011_q16_model(
        chordwise_element_count=FORMAL_Q16_GRID[0],
        spanwise_element_count=FORMAL_Q16_GRID[1],
        case=case,
    )
    structural = Q16CudaNewmarkStepper(
        model,
        boundary,
        device=config.DEVICE,
        newton_tolerance=3.0e-7,
        max_newton_iterations=128,
        cg_tolerance=2.0e-10,
        max_cg_iterations=2048,
        cg_check_every=16,
        nonsymmetric_solver="reference_dense",
        reference_dense_refresh_after=48,
        mass_damping_coefficient=0.0,
        stiffness_damping_coefficient=(
            case.structural_damping_loss_factor
            / (2.0 * math.pi * case.freestream_m_s / case.chord_m)
        ),
    )
    surface = Q16NativeV5MSurface(
        mesh,
        q16_chordwise_elements=FORMAL_Q16_GRID[0],
        q16_spanwise_elements=FORMAL_Q16_GRID[1],
        aerodynamic_chordwise_panels=FORMAL_AERO_GRID[0],
        aerodynamic_spanwise_panels=FORMAL_AERO_GRID[1],
        device=config.DEVICE,
        dense_transfers=True,
    )
    aerodynamic = Q16NativeV5MSolver(
        surface,
        NativeV5MConfig(
            chordwise_panels=FORMAL_AERO_GRID[0],
            spanwise_panels=FORMAL_AERO_GRID[1],
            density=case.fluid_density_kg_m3,
            freestream=case.freestream_m_s,
            aerodynamic_dt=case.aerodynamic_dt_s,
            lesp_crit=case.lesp_crit,
            wake_max_rows=300,
            particle_capacity=32768,
            particle_max_age_steps=100,
            wake_history_mode="bound_rate",
            wake_free_rows=100,
            dvm_target_spacing_chord=0.018,
            device=config.DEVICE,
        ),
    )
    state = wp.array(
        np.ascontiguousarray(mesh.reference_state[None, :]),
        dtype=config.DTYPE,
        device=config.DEVICE,
    )
    velocity = wp.zeros_like(state)
    acceleration = wp.zeros_like(state)
    owner = Q16NativeV5MFSIOwner.initialize(
        aerodynamic, state, velocity, acceleration
    )
    coupling = Q16NativeV5MFSIStepper(
        structural,
        aerodynamic,
        coupling_tolerance=5.0e-7,
        max_coupling_iterations=20,
        relaxation=0.7,
        persistent_relaxation=True,
        coupling_accelerator="aitken",
    )
    return case, surface, aerodynamic, owner, coupling


class TestMigrationParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")
        # Legacy path.
        cls.case, cls.surface, cls.aero, cls.owner, cls.coupling = (
            _legacy_stack()
        )
        cls._cn_normal_gpu = torch.tensor(
            plate_normal(cls.case), device="cuda:0", dtype=torch.float64
        )
        cls._cn_q_area = float(
            0.5
            * cls.case.fluid_density_kg_m3
            * cls.case.freestream_m_s**2
            * (cls.case.chord_m * cls.case.span_m)
        )
        cls.legacy = []
        substeps = 10
        prescribed = (None,) * substeps
        betas = tuple((i + 1) / substeps for i in range(substeps))
        for step in range(1, PARITY_STEPS + 1):
            time_star = step * cls.case.aerodynamic_dt_star
            apply_freestream(
                cls.aero, cls.case, cls.case.freestream_factor(time_star)
            )
            result = cls.coupling.advance(
                cls.owner,
                delta_time=cls.case.aerodynamic_dt_s,
                prescribed_forces=prescribed,
                load_betas=betas,
            )
            trial = result.aerodynamic.trial_state
            cls.legacy.append(
                {
                    "q": wp.to_torch(cls.owner.state).clone(),
                    "v": wp.to_torch(cls.owner.velocity).clone(),
                    "gamma_bound": trial.gamma_bound.clone(),
                    "wake_gamma": trial.wake_gamma.clone(),
                    "wake_rings": trial.wake_rings.clone(),
                    "pressure": result.aerodynamic.load.pressure.clone(),
                    "total_force": result.aerodynamic.load.total_force.clone(),
                    "particle_n": int(trial.particle_field.n),
                    # G000: both sides use the same GPU Cn projection
                    # operator; the historical CPU/numpy capture is retired
                    # so the parity gate stays bit-exact for ARCHITECTURE.
                    "cn": float(
                        (
                            torch.dot(
                                result.aerodynamic.load.total_force,
                                cls._cn_normal_gpu,
                            )
                            / cls._cn_q_area
                        ).item()
                    ),
                    "evaluations": result.aerodynamic_evaluations,
                    "iterations": result.coupling_iterations,
                    "digest": cls.owner.aerodynamic.state.digest(),
                }
            )

        # Unified path (same process, fresh stack).  The accelerator-cadence
        # knob is pinned to the legacy always-refresh value so this test
        # proves ARCHITECTURE parity bit-exactly; the drift-skip cadence is
        # validated separately (Newton-tolerance-level trajectory equivalence).
        runner = RojratsirikulCaseRunner(
            ROJ_A16_PRIMARY, reference_tangent_refresh_rtol=0.0
        )
        runner.build()
        payload = runner.run(max_aero_steps=PARITY_STEPS, output=None)
        cls.payload = payload
        cls.runner = runner
        import numpy as _np


    def test_step_count_and_status(self) -> None:
        self.assertEqual(self.payload["aero_steps"], PARITY_STEPS)
        self.assertEqual(
            self.payload["result_status"]["execution"], "completed"
        )
        self.assertEqual(
            self.payload["result_status"]["physics"], "passed"
        )

    def test_structural_state_bit_identical(self) -> None:
        # Per-step committed aero digests (they hash wake rings, gammas,
        # particles — everything) must match the legacy chain exactly.
        for index, legacy in enumerate(self.legacy):
            record = self.payload["records"][index]
            self.assertEqual(record["committed_digest"], legacy["digest"])
        final = self.legacy[-1]
        q_now = wp.to_torch(self.runner.owner.state)
        v_now = wp.to_torch(self.runner.owner.velocity)
        self.assertEqual(float((final["q"] - q_now).abs().max().item()), 0.0)
        self.assertEqual(float((final["v"] - v_now).abs().max().item()), 0.0)

    def test_aerodynamic_state_bit_identical(self) -> None:
        final = self.legacy[-1]
        trial = self.runner.owner.aerodynamic.state
        self.assertEqual(
            float(
                (final["gamma_bound"] - trial.gamma_bound).abs().max().item()
            ),
            0.0,
        )
        self.assertEqual(
            float((final["wake_gamma"] - trial.wake_gamma).abs().max().item()),
            0.0,
        )
        self.assertEqual(
            float((final["wake_rings"] - trial.wake_rings).abs().max().item()),
            0.0,
        )
        self.assertEqual(final["particle_n"], int(trial.particle_field.n))
        self.assertEqual(
            final["digest"], self.runner.owner.aerodynamic.state.digest()
        )

    def test_loads_and_counters_identical(self) -> None:
        for index, legacy in enumerate(self.legacy):
            record = self.payload["records"][index]
            self.assertEqual(legacy["cn"], record["cn"])
            self.assertEqual(
                legacy["evaluations"], record["aerodynamic_evaluations"]
            )
            self.assertEqual(
                legacy["iterations"], record["coupling_iterations"]
            )
            self.assertEqual(
                float(
                    np.abs(
                        np.asarray(legacy["total_force"].cpu().tolist())
                        - np.asarray(record["total_aerodynamic_force_n"])
                    ).max()
                ),
                0.0,
            )

    def test_unified_gate_records_present(self) -> None:
        first = self.payload["records"][0]
        self.assertTrue(
            first["unified_gate"]["unified_path_parent_digest_match"]
        )
        self.assertTrue(
            first["unified_gate"]["unified_path_parent_unchanged"]
        )
        for record in self.payload["records"]:
            self.assertEqual(
                record["unified_gate"]["formal_replay_count"], 1
            )
            self.assertEqual(record["unified_gate"]["commit_count"], 1)
            self.assertIn("particle_cull_count", record["aerodynamic"])
            self.assertIn("wake_truncate_ring_count", record["aerodynamic"])
            self.assertEqual(record["aerodynamic"]["release_owner_conflicts"], 0)

    def test_transfer_gates_close_exactly(self) -> None:
        gates = self.payload["transfer_gates"]
        self.assertTrue(gates["passed"])
        self.assertLessEqual(gates["max_errors"]["force_transfer_error"], 1e-12)
        self.assertLessEqual(gates["max_errors"]["moment_transfer_error"], 1e-12)
        self.assertLessEqual(gates["max_errors"]["virtual_work_error"], 1e-9)
        self.assertLessEqual(gates["max_errors"]["author_chain_error"], 1e-12)


if __name__ == "__main__":
    unittest.main()
