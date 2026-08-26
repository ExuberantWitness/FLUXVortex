"""U7-1: V5M3DStepper adapter — single-surface production parity.

Builds the full A16 production stack (frozen clock, ramped freestream,
dense transfers) twice, exactly as the production runner does:
1. driven through the unified AerodynamicStepper protocol
   (V5M3DStepper + Q16SurfaceFrameAdapter + set_structural_state bridge);
2. driven by direct native Q16NativeV5MSolver.propose / owner.commit.

Gate: bit-identical committed gamma_bound / wake / particle / diagnostics
state after every step, so the unified protocol cannot perturb the frozen
production numerics.

Also gates the U7-1 scope guards: multi-surface initialize/propose refuse
with a clear NotImplementedError, the transitional structural-state bridge
is required, frame topology is validated against the solver, and
FixedDynamics is the identity DynamicSubsystem for prescribed cases.
"""
from __future__ import annotations

import dataclasses
import math
import unittest
from types import SimpleNamespace

import numpy as np
import torch
import warp as wp

from fluxvortex.aero.protocol import AerodynamicStepper
from fluxvortex.aero.v5m.stepper import V5M3DStepper
from fluxvortex.dynamics.fixed_body import FixedDynamics
from fluxvortex.dynamics.protocol import DynamicSubsystem
from fluxvortex.kinematics.q16_surface import Q16SurfaceFrameAdapter
from fluxvortex.warp_fsi import config
from fluxvortex.warp_fsi.q16_flux_v5m_native import (
    NativeV5MConfig,
    Q16NativeV5MSolver,
    Q16NativeV5MSurface,
)
from fluxvortex.warp_fsi.q16_flux_v5m_native_fsi import Q16NativeV5MFSIOwner
from fluxvortex.warp_fsi.q16_structural_solver import Q16CudaNewmarkStepper
from forward_flight_benchmarks.rojratsirikul2011_q16 import (
    FORMAL_AERO_GRID,
    FORMAL_Q16_GRID,
    ROJ11_A16,
    make_rojratsirikul2011_q16_model,
)
from reproduce_rojratsirikul2011_q16_flux_v5m_native import _apply_freestream

OUTER_STEPS = 3
# ~3 chords of free wake + one-chord LEV particle lifetime (runner freeze).
WAKE_MAX_ROWS = 300
PARTICLE_CAPACITY = 32768
PARTICLE_MAX_AGE_STEPS = 100
AERO_DT = ROJ11_A16.aerodynamic_dt_s


def _build_production_stack() -> SimpleNamespace:
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
    return SimpleNamespace(
        structural=structural,
        surface=surface,
        aerodynamic=aerodynamic,
        owner=owner,
        state=state,
        velocity=velocity,
    )


@unittest.skipUnless(
    torch.cuda.is_available() and wp.is_cuda_available(), "CUDA required"
)
class U7V5MStepperAdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.unified = _build_production_stack()
        cls.direct = _build_production_stack()
        cls.kinematics = Q16SurfaceFrameAdapter(cls.unified.surface)
        cls.frame = cls.kinematics.evaluate(cls.unified.state, cls.unified.velocity)
        cls.stepper = V5M3DStepper(cls.unified.aerodynamic, device=config.DEVICE)
        cls.second_frame = dataclasses.replace(cls.frame, surface_id="wing_1")

    def test_01_protocol_conformance_and_topology(self):
        """The adapter satisfies AerodynamicStepper and builds the topology."""
        self.assertIsInstance(self.stepper, AerodynamicStepper)
        returned = self.stepper.initialize((self.frame,))
        self.assertIs(returned, self.unified.aerodynamic)
        topology = self.stepper.topology
        self.assertIsNotNone(topology)
        self.assertEqual(len(topology.surfaces), 1)
        offsets = topology.surfaces[0]
        self.assertEqual(offsets.surface_id, "wing_0")
        self.assertEqual(offsets.panel_offset, 0)
        self.assertEqual(
            offsets.panel_count, FORMAL_AERO_GRID[0] * FORMAL_AERO_GRID[1]
        )
        self.assertEqual(offsets.strip_count, FORMAL_AERO_GRID[1])
        self.assertEqual(offsets.le_node_count, FORMAL_AERO_GRID[1] + 1)
        self.assertEqual(
            topology.total_panels, FORMAL_AERO_GRID[0] * FORMAL_AERO_GRID[1]
        )
        self.assertEqual(topology.surface_of_panel(0), "wing_0")
        self.assertEqual(topology.global_panel_index("wing_0", 0), 0)
        # frame metadata is carried through the kinematics adapter
        self.assertEqual(self.frame.chordwise_panels, FORMAL_AERO_GRID[0])
        self.assertEqual(self.frame.spanwise_panels, FORMAL_AERO_GRID[1])

    def test_02_single_surface_production_parity(self):
        """Unified propose/commit is bit-identical to direct native stepping."""
        self.stepper.initialize((self.frame,))
        self.stepper.set_structural_state(self.unified.state, self.unified.velocity)
        for step in range(1, OUTER_STEPS + 1):
            factor = ROJ11_A16.freestream_factor(step * ROJ11_A16.aerodynamic_dt_star)
            _apply_freestream(self.unified.aerodynamic, ROJ11_A16, factor)
            _apply_freestream(self.direct.aerodynamic, ROJ11_A16, factor)
            # unified protocol path
            proposal = self.stepper.propose(
                self.unified.owner, (self.frame,), AERO_DT
            )
            self.stepper.commit(self.unified.owner, proposal)
            # direct native path (the frozen production call)
            direct_proposal = self.direct.aerodynamic.propose(
                self.direct.owner.aerodynamic.state,
                self.direct.state,
                self.direct.velocity,
            )
            self.direct.owner.aerodynamic.commit(direct_proposal)
            unified_state = self.unified.owner.aerodynamic.state
            direct_state = self.direct.owner.aerodynamic.state
            # committed bound circulation bit-identical
            self.assertTrue(torch.equal(
                unified_state.gamma_bound, direct_state.gamma_bound
            ))
            self.assertEqual(unified_state.step, step)
            self.assertEqual(direct_state.step, step)
            # full committed state agrees (wake, particles, frontier)
            self.assertEqual(unified_state.digest(), direct_state.digest())
            self.assertTrue(torch.equal(
                unified_state.wake_gamma, direct_state.wake_gamma
            ))
            self.assertEqual(
                unified_state.particle_field.n, direct_state.particle_field.n
            )
            # proposals agree bit-identically (loads + generalized force)
            self.assertTrue(torch.equal(
                proposal.load.pressure, direct_proposal.load.pressure
            ))
            self.assertTrue(torch.equal(
                proposal.load.panel_forces, direct_proposal.load.panel_forces
            ))
            self.assertTrue(torch.equal(
                wp.to_torch(proposal.generalized_force),
                wp.to_torch(direct_proposal.generalized_force),
            ))
            self.assertEqual(
                unified_state.diagnostics[-1], direct_state.diagnostics[-1]
            )

    def test_03_multi_surface_rejected(self):
        """Two frames refuse with a clear U7-1 scope error."""
        with self.assertRaises(NotImplementedError) as context:
            self.stepper.initialize((self.frame, self.second_frame))
        message = str(context.exception)
        self.assertIn("Multi-surface", message)
        self.assertIn("2 surfaces", message)
        self.assertIn("U7-1", message)
        # the topology bookkeeping was still built for the refused layout
        self.assertEqual(
            self.stepper.topology.total_panels,
            2 * FORMAL_AERO_GRID[0] * FORMAL_AERO_GRID[1],
        )
        # propose refuses multi-surface before touching the solver
        self.stepper.set_structural_state(self.unified.state, self.unified.velocity)
        with self.assertRaises(NotImplementedError) as propose_context:
            self.stepper.propose(
                self.unified.owner, (self.frame, self.second_frame), AERO_DT
            )
        self.assertIn("U7+", str(propose_context.exception))
        # neither refusal advanced the committed aero state
        self.assertEqual(self.unified.owner.aerodynamic.state.step, OUTER_STEPS)

    def test_04_structural_state_bridge_required(self):
        """propose() without the transitional bridge fails loudly."""
        fresh = V5M3DStepper(self.unified.aerodynamic, device=config.DEVICE)
        fresh.initialize((self.frame,))
        with self.assertRaises(RuntimeError) as context:
            fresh.propose(self.unified.owner, (self.frame,), AERO_DT)
        self.assertIn("set_structural_state", str(context.exception))
        # and initialize() itself must precede propose()
        uninitialized = V5M3DStepper(self.unified.aerodynamic, device=config.DEVICE)
        uninitialized.set_structural_state(self.unified.state, self.unified.velocity)
        with self.assertRaises(RuntimeError) as init_context:
            uninitialized.propose(self.unified.owner, (self.frame,), AERO_DT)
        self.assertIn("initialize", str(init_context.exception))

    def test_05_initialize_validates_topology(self):
        """A frame whose panel counts differ from the solver is rejected."""
        self.stepper.set_structural_state(self.unified.state, self.unified.velocity)
        mismatched = dataclasses.replace(
            self.frame, chordwise_panels=self.frame.chordwise_panels + 1
        )
        with self.assertRaises(ValueError):
            self.stepper.initialize((mismatched,))
        with self.assertRaises(ValueError):
            self.stepper.initialize(())


class FixedDynamicsTest(unittest.TestCase):
    """U7-1: FixedDynamics is the identity DynamicSubsystem (CPU)."""

    def test_protocol_conformance(self):
        self.assertIsInstance(FixedDynamics(), DynamicSubsystem)

    def test_predict_is_identity(self):
        dynamics = FixedDynamics()
        committed = object()
        self.assertIs(dynamics.predict(committed, AERO_DT), committed)

    def test_propose_is_identity(self):
        dynamics = FixedDynamics()
        committed = object()
        self.assertIs(dynamics.propose(committed, None, AERO_DT), committed)


if __name__ == "__main__":
    unittest.main()
