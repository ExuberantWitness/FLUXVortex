"""U1: unified FSI transaction — parity with production stepper + counters.

Runs the formal A16 grid (5x10 Q16, 15x30 aero) for 3 outer steps:
1. Production stepper result vs wrapper result bit-identical structural state.
2. Transaction counters have true semantics (1 commit, 1 formal replay, N proposals).
3. GlobalTransaction rejects double commit.
4. Failed trial leaves owner digest unchanged.

The two GPU stacks (bare production stepper vs the same stepper wrapped in
PartitionedStrongFSI) are built independently with the exact production
setup of reproduce_rojratsirikul2011_q16_flux_v5m_native.py (frozen clock,
coupling tolerance, relaxation, ramped freestream), so any difference in
the committed states is attributable to the wrapper alone.
"""
from __future__ import annotations

import dataclasses
import math
import unittest

import numpy as np
import torch
import warp as wp

from fluxvortex.coupling.partitioned import (
    PartitionedStrongFSI,
    TransactionCounters,
)
from fluxvortex.dynamics.q16_adapter import Q16DynamicsAdapter
from fluxvortex.state.transaction import GlobalTransaction
from fluxvortex.state.world import WorldDynamicState
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
    ROJ11_A16,
    make_rojratsirikul2011_q16_model,
)
from reproduce_rojratsirikul2011_q16_flux_v5m_native import _apply_freestream

OUTER_STEPS = 3
# Frozen production clock / coupling (reproduce_..._native.py + case freeze).
SUBSTEPS = ROJ11_A16.structural_substeps_per_aerodynamic_step
COUPLING_TOLERANCE = 5.0e-7
RELAXATION = 0.7
MAX_COUPLING_ITERATIONS = 20
# ~3 chords of free wake + one-chord LEV particle lifetime (runner freeze).
WAKE_MAX_ROWS = 300
PARTICLE_CAPACITY = 32768
PARTICLE_MAX_AGE_STEPS = 100

PRESCRIBED = (None,) * SUBSTEPS
LOAD_BETAS = tuple((index + 1) / SUBSTEPS for index in range(SUBSTEPS))


def _build_production_stack() -> tuple[Q16NativeV5MFSIOwner,
                                       Q16NativeV5MFSIStepper,
                                       Q16NativeV5MSolver]:
    """Build one independent A16 stack exactly as the production runner does."""

    mesh, model, boundary, _ = make_rojratsirikul2011_q16_model(
        chordwise_element_count=FORMAL_Q16_GRID[0],
        spanwise_element_count=FORMAL_Q16_GRID[1],
        case=ROJ11_A16,
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
            ROJ11_A16.structural_damping_loss_factor
            / (2.0 * math.pi * ROJ11_A16.freestream_m_s / ROJ11_A16.chord_m)
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
            density=ROJ11_A16.fluid_density_kg_m3,
            freestream=ROJ11_A16.freestream_m_s,
            aerodynamic_dt=ROJ11_A16.aerodynamic_dt_s,
            lesp_crit=ROJ11_A16.lesp_crit,
            wake_max_rows=WAKE_MAX_ROWS,
            particle_capacity=PARTICLE_CAPACITY,
            particle_max_age_steps=PARTICLE_MAX_AGE_STEPS,
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
    owner = Q16NativeV5MFSIOwner.initialize(aerodynamic, state, velocity, acceleration)
    stepper = Q16NativeV5MFSIStepper(
        structural,
        aerodynamic,
        coupling_tolerance=COUPLING_TOLERANCE,
        max_coupling_iterations=MAX_COUPLING_ITERATIONS,
        relaxation=RELAXATION,
        persistent_relaxation=True,
        coupling_accelerator="aitken",
    )
    return owner, stepper, aerodynamic


class _Elastic:
    """ElasticState-like holder for the unified WorldDynamicState."""

    def __init__(self, q: wp.array, qd: wp.array, qdd: wp.array) -> None:
        self.q, self.qd, self.qdd = q, qd, qdd


class GlobalTransactionGuardTest(unittest.TestCase):
    """U1: GlobalTransaction one-commit-per-step guard (CPU)."""

    def test_double_commit_rejected(self):
        transaction = GlobalTransaction()
        transaction.begin_step(0)
        transaction.commit(0)
        with self.assertRaises(RuntimeError):
            transaction.commit(0)

    def test_rebegin_committed_step_rejected(self):
        transaction = GlobalTransaction()
        transaction.begin_step(0)
        transaction.commit(0)
        with self.assertRaises(RuntimeError):
            transaction.begin_step(0)

    def test_commit_without_begin_step_rejected(self):
        transaction = GlobalTransaction()
        with self.assertRaises(RuntimeError):
            transaction.commit(3)

    def test_next_step_allows_exactly_one_commit(self):
        transaction = GlobalTransaction()
        for step in (0, 1, 2):
            transaction.begin_step(step)
            transaction.commit(step)


@unittest.skipUnless(
    torch.cuda.is_available() and wp.is_cuda_available(), "CUDA required"
)
class U1FSITransactionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bare_owner, cls.bare_stepper, cls.bare_aero = _build_production_stack()
        (
            cls.wrapped_owner,
            wrapped_stepper,
            cls.wrapped_aero,
        ) = _build_production_stack()
        cls.wrapper = PartitionedStrongFSI(wrapped_stepper)

    def _advance_bare(self, step: int):
        _apply_freestream(
            self.bare_aero, ROJ11_A16,
            ROJ11_A16.freestream_factor(step * ROJ11_A16.aerodynamic_dt_star),
        )
        return self.bare_stepper.advance(
            self.bare_owner,
            delta_time=ROJ11_A16.aerodynamic_dt_s,
            prescribed_forces=PRESCRIBED,
            load_betas=LOAD_BETAS,
        )

    def _advance_wrapped(self, step: int):
        _apply_freestream(
            self.wrapped_aero, ROJ11_A16,
            ROJ11_A16.freestream_factor(step * ROJ11_A16.aerodynamic_dt_star),
        )
        return self.wrapper.advance(
            self.wrapped_owner,
            delta_time=ROJ11_A16.aerodynamic_dt_s,
            prescribed_forces=PRESCRIBED,
            load_betas=LOAD_BETAS,
        )

    def test_01_parity_and_true_transaction_counters(self):
        """Wrapper is bit-identical to the production stepper and counts truth."""
        bare_evaluations = 0
        for step in range(1, OUTER_STEPS + 1):
            bare = self._advance_bare(step)
            wrapped = self._advance_wrapped(step)
            bare_evaluations += bare.aerodynamic_evaluations
            # (1) committed structural state bit-identical after each step
            self.assertTrue(torch.equal(
                wp.to_torch(self.bare_owner.state),
                wp.to_torch(self.wrapped_owner.state),
            ))
            self.assertTrue(torch.equal(
                wp.to_torch(self.bare_owner.velocity),
                wp.to_torch(self.wrapped_owner.velocity),
            ))
            self.assertTrue(torch.equal(
                wp.to_torch(self.bare_owner.acceleration),
                wp.to_torch(self.wrapped_owner.acceleration),
            ))
            # returned step result identical as well
            for field in ("state", "velocity", "acceleration"):
                self.assertTrue(torch.equal(
                    wp.to_torch(getattr(bare.structural, field)),
                    wp.to_torch(getattr(wrapped.structural, field)),
                ))
            # committed bound circulation identical (aero side parity)
            self.assertTrue(torch.equal(
                self.bare_owner.aerodynamic.state.gamma_bound,
                self.wrapped_owner.aerodynamic.state.gamma_bound,
            ))
            self.assertEqual(self.bare_owner.generation, step)
            self.assertEqual(self.wrapped_owner.generation, step)
            # at least one exploratory trial plus one formal proposal per step
            self.assertGreaterEqual(bare.aerodynamic_evaluations, 2)
            # (2) true-semantics counters after `step` accepted steps
            counters = self.wrapper.counters
            self.assertIsInstance(counters, TransactionCounters)
            self.assertEqual(counters.commit_count, step)
            self.assertEqual(counters.formal_replay_count, step)
            self.assertGreaterEqual(counters.discarded_trial_count, 0)
            self.assertGreaterEqual(counters.aero_proposal_count, step)
            self.assertEqual(
                counters.aero_proposal_count,
                counters.discarded_trial_count + step,
            )
        # accounting totals match the bare production stepper exactly
        self.assertEqual(self.wrapper.counters.aero_proposal_count, bare_evaluations)
        self.assertEqual(self.wrapper.counters.commit_count, OUTER_STEPS)
        self.assertEqual(self.wrapper.counters.formal_replay_count, OUTER_STEPS)
        self.assertEqual(
            self.wrapper.counters.discarded_trial_count, bare_evaluations - OUTER_STEPS
        )

    def test_02_failed_trial_leaves_owner_unchanged(self):
        """A rejected advance must not touch owner generation, digest, or state."""
        generation = self.wrapped_owner.generation
        digest = self.wrapped_owner.aerodynamic.state.digest()
        state = wp.to_torch(self.wrapped_owner.state).clone()
        counters = dataclasses.replace(self.wrapper.counters)
        # invalid delta_time (non-positive) -> stepper's own validation
        with self.assertRaises(ValueError):
            self.wrapper.advance(
                self.wrapped_owner,
                delta_time=0.0,
                prescribed_forces=PRESCRIBED,
                load_betas=LOAD_BETAS,
            )
        # mismatched clock (double the frozen aerodynamic dt)
        with self.assertRaises(ValueError):
            self.wrapper.advance(
                self.wrapped_owner,
                delta_time=2.0 * ROJ11_A16.aerodynamic_dt_s,
                prescribed_forces=PRESCRIBED,
                load_betas=LOAD_BETAS,
            )
        self.assertEqual(self.wrapped_owner.generation, generation)
        self.assertEqual(self.wrapped_owner.aerodynamic.state.digest(), digest)
        self.assertTrue(torch.equal(wp.to_torch(self.wrapped_owner.state), state))
        # the failed trials must not be counted as commits or proposals
        self.assertEqual(self.wrapper.counters, counters)

    def test_03_dynamics_adapter_wraps_production_solver(self):
        """Q16DynamicsAdapter delegates bit-identically to the CUDA stepper."""
        solver = self.wrapper._stepper.structural_solver
        elastic = _Elastic(
            self.wrapped_owner.state,
            self.wrapped_owner.velocity,
            self.wrapped_owner.acceleration,
        )
        committed = WorldDynamicState(
            elastic_states={"membrane_0": elastic}, body_states={}, joint_states={}
        )
        adapter = Q16DynamicsAdapter(solver)
        dt = ROJ11_A16.structural_dt_s
        q_pred, v_pred = adapter.predict(committed, dt)
        ref_q, ref_v = solver.predict_kinematics(
            self.wrapped_owner.state,
            self.wrapped_owner.velocity,
            self.wrapped_owner.acceleration,
            delta_time=dt,
        )
        self.assertTrue(torch.equal(wp.to_torch(q_pred), wp.to_torch(ref_q)))
        self.assertTrue(torch.equal(wp.to_torch(v_pred), wp.to_torch(ref_v)))
        zero_load = wp.zeros_like(self.wrapped_owner.state)
        proposal = adapter.propose(committed, zero_load, dt)
        reference = solver.step(
            self.wrapped_owner.state,
            self.wrapped_owner.velocity,
            self.wrapped_owner.acceleration,
            zero_load,
            delta_time=dt,
        )
        for field in ("state", "velocity", "acceleration"):
            self.assertTrue(torch.equal(
                wp.to_torch(getattr(proposal, field)),
                wp.to_torch(getattr(reference, field)),
            ))
        # the adapter never mutated the committed owner arrays
        self.assertEqual(self.wrapped_owner.generation, OUTER_STEPS)


if __name__ == "__main__":
    unittest.main()
