"""U2: unified V5M state, circulation/impulse ledger, retention policies.

1. V5MWorldState.from_native_state wraps the A16 production solver's state by
   reference (bound circulation, wake, particle field identity).
2. AgeCapPolicy removes exactly the aged-out particles and accounts the
   removed circulation/linear impulse in a RetentionResult.
3. WakeRowCapPolicy flags excess oldest rows with their circulation.
4. NoCulling never removes anything.
5. CirculationImpulseLedger holds the per-step audit quantities (§5.9).

The A16 stack is built with the exact production setup of
reproduce_rojratsirikul2011_q16_flux_v5m_native.py (frozen grids, Lcrit,
caps), mirroring the U1 test pattern.
"""
from __future__ import annotations

import unittest

import numpy as np
import torch
import warp as wp

from fluxvortex.aero.v5m.retention import (
    AgeCapPolicy,
    NoCulling,
    RetentionResult,
    WakeRowCapPolicy,
)
from fluxvortex.aero.v5m.state import (
    CirculationImpulseLedger,
    V5MSurfaceState,
    V5MWorldState,
)
from fluxvortex.warp_fsi import config
from fluxvortex.warp_fsi.q16_flux_v5m_native import (
    NativeV5MConfig,
    Q16NativeV5MSolver,
    Q16NativeV5MSurface,
)
from forward_flight_benchmarks.rojratsirikul2011_q16 import (
    FORMAL_AERO_GRID,
    FORMAL_Q16_GRID,
    ROJ11_A16,
    make_rojratsirikul2011_q16_model,
)
from pfield_torch_gpu import CudaParticleField

# Same production caps as the U1 test (runner freeze).
WAKE_MAX_ROWS = 300
PARTICLE_CAPACITY = 32768
PARTICLE_MAX_AGE_STEPS = 100


def _build_production_aero() -> Q16NativeV5MSolver:
    """Build the A16 production aero solver exactly as the runner does."""
    mesh, _, _, _ = make_rojratsirikul2011_q16_model(
        chordwise_element_count=FORMAL_Q16_GRID[0],
        spanwise_element_count=FORMAL_Q16_GRID[1],
        case=ROJ11_A16,
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
    return Q16NativeV5MSolver(
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


def _reference_arrays(solver: Q16NativeV5MSolver):
    """Flat (state, velocity) warp arrays at the mesh reference state."""
    mesh = solver.surface.mesh
    state = wp.array(
        np.ascontiguousarray(mesh.reference_state[None, :]),
        dtype=config.DTYPE,
        device=config.DEVICE,
    )
    velocity = wp.zeros_like(state)
    return state, velocity


def _make_field_with_batches(
    batches: list[tuple[int, torch.Tensor, torch.Tensor]],
) -> CudaParticleField:
    """Field holding (birth_step, pos, gamma) batches added in order."""
    device = torch.device(config.DEVICE)
    capacity = 4 * sum(pos.shape[0] for _, pos, _ in batches)
    field = CudaParticleField(capacity, device=device)
    for birth_step, pos, gamma in batches:
        sigma = torch.full(
            (pos.shape[0],), 1.0e-3, device=device, dtype=torch.float64
        )
        field.add_particles(pos, gamma, sigma, birth_step=birth_step)
    return field


@unittest.skipUnless(
    torch.cuda.is_available() and wp.is_cuda_available(), "CUDA required"
)
class U2WorldStateFromNativeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.aero = _build_production_aero()
        state, velocity = _reference_arrays(cls.aero)
        cls.native = cls.aero.initialize(state, velocity)

    def test_01_from_native_state_wraps_by_reference(self):
        world = V5MWorldState.from_native_state(self.native)
        self.assertEqual(world.step_index, self.native.step)
        self.assertEqual(world.step_index, 0)
        self.assertEqual(world.generation, 0)
        self.assertIsNone(world.ledger)
        self.assertEqual(set(world.surfaces), {"wing_0"})
        surface = world.surfaces["wing_0"]
        # wrapped tensors are the production solver's own truth (by reference)
        self.assertTrue(torch.equal(surface.bound_circulation, self.native.gamma_bound))
        self.assertTrue(
            torch.equal(surface.previous_bound_circulation, self.native.gamma_previous)
        )
        self.assertTrue(torch.equal(surface.wake_rings, self.native.wake_rings))
        self.assertTrue(torch.equal(surface.wake_gamma, self.native.wake_gamma))
        self.assertIs(surface.particle_field, self.native.particle_field)
        # separation truth starts unowned in the wrapper; the solver's propose
        # is the single owner that fills it (plan §8.3)
        self.assertIsNone(surface.lesp_pre_3d)
        self.assertIsNone(surface.surface_separated)
        self.assertEqual(surface.lesp_crit, ROJ11_A16.lesp_crit)
        # shapes/topology match the frozen A16 grid
        n_panels = FORMAL_AERO_GRID[0] * FORMAL_AERO_GRID[1]
        self.assertEqual(surface.bound_circulation.shape, (n_panels,))
        self.assertEqual(surface.wake_rings.shape[0], 0)
        self.assertEqual(surface.particle_field.n, 0)

    def test_02_custom_surface_id_and_lesp_crit(self):
        world = V5MWorldState.from_native_state(
            self.native, surface_id="left_wing", lesp_crit=0.13
        )
        self.assertEqual(set(world.surfaces), {"left_wing"})
        self.assertEqual(world.surfaces["left_wing"].lesp_crit, 0.13)


@unittest.skipUnless(
    torch.cuda.is_available() and wp.is_cuda_available(), "CUDA required"
)
class U2RetentionPolicyTest(unittest.TestCase):
    def _batches(self):
        device = torch.device(config.DEVICE)
        old_pos = torch.tensor(
            [[1.0, 0.1, 0.0], [1.0, 0.2, 0.0], [1.0, 0.3, 0.0]],
            device=device, dtype=torch.float64,
        )
        old_gamma = torch.tensor(
            [[0.3, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, -0.25]],
            device=device, dtype=torch.float64,
        )
        mid_pos = torch.tensor(
            [[0.9, 0.15, 0.0], [0.9, 0.25, 0.0]], device=device, dtype=torch.float64
        )
        mid_gamma = torch.tensor(
            [[0.1, 0.0, 0.0], [0.0, -0.15, 0.0]], device=device, dtype=torch.float64
        )
        new_pos = torch.tensor(
            [[0.8, 0.1, 0.05], [0.8, 0.2, 0.05], [0.8, 0.3, 0.05], [0.8, 0.4, 0.05]],
            device=device, dtype=torch.float64,
        )
        new_gamma = 0.05 * torch.ones(
            (4, 3), device=device, dtype=torch.float64
        )
        return [
            (5, old_pos, old_gamma),
            (15, mid_pos, mid_gamma),
            (20, new_pos, new_gamma),
        ]

    def test_01_age_cap_removes_old_and_accounts(self):
        field = _make_field_with_batches(self._batches())
        self.assertEqual(field.n, 9)
        policy = AgeCapPolicy(max_age_steps=10)
        # step 25, horizon 15: birth 5 is aged out; birth 15/20 survive
        # (the production mask is birth < horizon, strictly).
        result = policy.apply_particles(field, step=25)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, RetentionResult)
        self.assertEqual(result.removed_particle_count, 3)
        expected_circ = float(
            torch.tensor([0.3, 0.2, 0.25], device=field.device).sum().item()
        )
        self.assertAlmostEqual(result.removed_circulation_sum, expected_circ, places=14)
        self.assertGreater(result.removed_circulation_sum, 0.0)
        # linear impulse audit of the removed set is a finite (3,) CUDA vector
        self.assertIsNotNone(result.removed_linear_impulse)
        lin = result.removed_linear_impulse
        self.assertEqual(tuple(lin.shape), (3,))
        self.assertTrue(lin.is_cuda)
        self.assertTrue(bool(torch.isfinite(lin).all().item()))
        self.assertIsNone(result.removed_angular_impulse)
        self.assertEqual(result.conservation_error, 0.0)
        # removal actually happened: only the 15/20 cohorts remain, in order
        self.assertEqual(field.n, 6)
        self.assertEqual(
            field.birth_step[: field.n].tolist(), [15, 15, 20, 20, 20, 20]
        )
        self.assertTrue(
            torch.equal(field.pos[: field.n][2:], self._batches()[2][1])
        )
        # applying again is a no-op (nothing older than the horizon left)
        self.assertIsNone(policy.apply_particles(field, step=25))

    def test_02_age_cap_noop_cases(self):
        # empty field
        empty = CudaParticleField(16, device=torch.device(config.DEVICE))
        policy = AgeCapPolicy(max_age_steps=10)
        self.assertIsNone(policy.apply_particles(empty, step=1000))
        # horizon not reached yet (step <= max_age)
        field = _make_field_with_batches(self._batches())
        self.assertIsNone(policy.apply_particles(field, step=10))
        self.assertEqual(field.n, 9)
        # horizon positive but no particle is old enough: step 15 -> horizon 5,
        # and the oldest cohort was born exactly at 5 (mask is strict <).
        self.assertIsNone(policy.apply_particles(field, step=15))
        self.assertEqual(field.n, 9)

    def test_03_wake_row_cap_flags_excess(self):
        device = torch.device(config.DEVICE)
        ns = 10
        rows = 8
        wake_rings = 0.01 * torch.arange(rows * ns * 4 * 3, device=device, dtype=torch.float64).reshape(rows * ns, 4, 3)
        wake_gamma = 0.001 * torch.arange(rows * ns, device=device, dtype=torch.float64) - 0.02
        policy = WakeRowCapPolicy(max_rows=5, spanwise_panels=ns)
        result = policy.apply_wake(wake_rings, wake_gamma, max_rows=5)
        self.assertIsNotNone(result)
        excess = rows * ns - 5 * ns
        self.assertEqual(result.removed_particle_count, 0)
        expected = float(wake_gamma[rows * ns - excess :].abs().sum().item())
        self.assertAlmostEqual(result.removed_circulation_sum, expected, places=14)
        self.assertGreater(result.removed_circulation_sum, 0.0)
        self.assertIsNone(result.removed_linear_impulse)
        self.assertIsNone(result.removed_angular_impulse)
        # under the cap: nothing to flag
        self.assertIsNone(
            policy.apply_wake(wake_rings[: 3 * ns], wake_gamma[: 3 * ns], max_rows=5)
        )
        # exactly at the cap: nothing to flag
        self.assertIsNone(
            policy.apply_wake(wake_rings[: 5 * ns], wake_gamma[: 5 * ns], max_rows=5)
        )

    def test_04_no_culling_never_removes(self):
        no_culling = NoCulling()
        field = _make_field_with_batches(self._batches())
        before = field.birth_step[: field.n].clone()
        self.assertIsNone(no_culling.apply_particles(field, step=10_000))
        self.assertEqual(field.n, before.shape[0])
        self.assertTrue(torch.equal(field.birth_step[: field.n], before))
        wake_gamma = torch.ones(400, device=field.device, dtype=torch.float64)
        wake_rings = torch.zeros((400, 4, 3), device=field.device, dtype=torch.float64)
        self.assertIsNone(no_culling.apply_wake(wake_rings, wake_gamma, max_rows=1))
        self.assertEqual(wake_gamma.numel(), 400)


@unittest.skipUnless(
    torch.cuda.is_available() and wp.is_cuda_available(), "CUDA required"
)
class U2CirculationImpulseLedgerTest(unittest.TestCase):
    def test_01_ledger_construction_and_defaults(self):
        device = torch.device(config.DEVICE)
        bound_before = torch.ones(150, device=device, dtype=torch.float64)
        bound_after = 0.5 * bound_before
        ledger = CirculationImpulseLedger(
            step=7,
            bound_before=bound_before,
            bound_after=bound_after,
            newborn_lev_circulation=12.5,
            newborn_tev_circulation=-3.25,
            removed_circulation=0.75,
            kelvin_residual=1.0e-9,
        )
        self.assertEqual(ledger.step, 7)
        self.assertTrue(torch.equal(ledger.bound_before, bound_before))
        self.assertTrue(torch.equal(ledger.bound_after, bound_after))
        self.assertEqual(ledger.newborn_lev_circulation, 12.5)
        self.assertEqual(ledger.newborn_tev_circulation, -3.25)
        self.assertEqual(ledger.removed_circulation, 0.75)
        self.assertIsNone(ledger.linear_impulse_before)
        self.assertIsNone(ledger.linear_impulse_after)
        self.assertIsNone(ledger.angular_impulse_before)
        self.assertIsNone(ledger.angular_impulse_after)
        # the conservation-error property currently proxies the Kelvin residual
        self.assertEqual(ledger.circulation_conservation_error, 1.0e-9)

    def test_02_ledger_with_impulse_terms(self):
        device = torch.device(config.DEVICE)
        zeros3 = torch.zeros(3, device=device, dtype=torch.float64)
        ledger = CirculationImpulseLedger(
            step=1,
            bound_before=zeros3[:1].repeat(10),
            bound_after=zeros3[:1].repeat(10),
            newborn_lev_circulation=0.0,
            newborn_tev_circulation=0.0,
            removed_circulation=0.0,
            kelvin_residual=0.0,
            linear_impulse_before=zeros3.clone(),
            linear_impulse_after=zeros3.clone(),
            angular_impulse_before=zeros3.clone(),
            angular_impulse_after=zeros3.clone(),
        )
        for name in (
            "linear_impulse_before",
            "linear_impulse_after",
            "angular_impulse_before",
            "angular_impulse_after",
        ):
            value = getattr(ledger, name)
            self.assertTrue(value.is_cuda)
            self.assertEqual(tuple(value.shape), (3,))
        self.assertEqual(ledger.circulation_conservation_error, 0.0)

    def test_03_ledger_attaches_to_world_state(self):
        device = torch.device(config.DEVICE)
        field = CudaParticleField(16, device=device)
        surface = V5MSurfaceState(
            surface_id="wing_0",
            bound_circulation=torch.zeros(10, device=device, dtype=torch.float64),
            previous_bound_circulation=torch.zeros(10, device=device, dtype=torch.float64),
            lesp_pre_3d=None,
            surface_separated=None,
            lesp_crit=0.11,
            particle_field=field,
            wake_rings=torch.zeros((0, 4, 3), device=device, dtype=torch.float64),
            wake_gamma=torch.zeros(0, device=device, dtype=torch.float64),
        )
        ledger = CirculationImpulseLedger(
            step=3,
            bound_before=surface.bound_circulation,
            bound_after=surface.bound_circulation.clone(),
            newborn_lev_circulation=0.0,
            newborn_tev_circulation=0.0,
            removed_circulation=0.0,
            kelvin_residual=0.0,
        )
        world = V5MWorldState(
            step_index=3, generation=1, surfaces={"wing_0": surface}, ledger=ledger
        )
        self.assertIs(world.ledger, ledger)
        self.assertIs(world.surfaces["wing_0"], surface)


if __name__ == "__main__":
    unittest.main()
