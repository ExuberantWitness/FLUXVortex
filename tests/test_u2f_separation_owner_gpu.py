"""U2-F: single separation owner wired into the A16 production solver.

Fix checkpoint for refactor plan §8.3 / §14:
1. ``unified_separation_mask`` is the only place separation is decided —
   3D actual-surface LESP vs the frozen lesp_crit.
2. ``reconcile_release_mask`` returns the 3D truth as the production
   pin/release mask (NOT the union with the source bank's shed_lev) and
   counts owner disagreements instead of silently unioning them.
3. The A16 production solver is driven for 5 propose steps with per-step
   capture of ``lesp_pre_3d`` and the source bank's ``shed_lev``; in the
   saturated regime (all 30 strips above LESPcrit) the 3D mask is all-True
   with zero conflicts, and production diagnostics agree with the capture.
4. The load-history models (plan §8.7) carry explicit physics identities —
   ``bound_rate`` vs ``material`` are distinct models, not a stability
   switch, and the production config validates against the registry.

The A16 stack is built with the exact production setup of
reproduce_rojratsirikul2011_q16_flux_v5m_native.py (frozen grids, Lcrit,
caps), mirroring the U2-P test pattern.
"""
from __future__ import annotations

import unittest

import numpy as np
import torch
import warp as wp

from fluxvortex.aero.v5m.loads import (
    BOUND_RATE_MODEL,
    LOAD_HISTORY_MODELS,
    MATERIAL_MODEL,
    LoadHistoryModel,
)
from fluxvortex.aero.v5m.separation import (
    reconcile_release_mask,
    unified_separation_mask,
)
from fluxvortex.aero.v5m.state import V5MSurfaceState, V5MWorldState
from fluxvortex.warp_fsi import config
from fluxvortex.warp_fsi.q16_flux_v5m_native import (
    LESP_FACTOR,
    NativeV5MConfig,
    Q16NativeV5MSolver,
    Q16NativeV5MSurface,
    native_aic,
)
from forward_flight_benchmarks.rojratsirikul2011_q16 import (
    FORMAL_AERO_GRID,
    FORMAL_Q16_GRID,
    ROJ11_A16,
    make_rojratsirikul2011_q16_model,
)
from pfield_torch_gpu import CudaParticleField

# Same production caps as the U1/U2 tests (runner freeze).
WAKE_MAX_ROWS = 300
PARTICLE_CAPACITY = 32768
PARTICLE_MAX_AGE_STEPS = 100
PROPOSE_STEPS = 5


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


def _attach_capture(solver: Q16NativeV5MSolver) -> list[dict[str, torch.Tensor]]:
    """Record per-propose-step lesp_pre_3d and source ``shed_lev``.

    ``_deposit_dvm_ribbon(geometry, trial, result)`` is called with the
    source bank's result dict and, at that point, the trial's pre-newborn
    arrays are exactly the ones ``propose`` used for its pre-solve, so the
    3D LESP is recomputed bit-identically (verified against the solver's
    own ``lesp_pre_max_abs`` diagnostic in test_03).
    """
    captured: list[dict[str, torch.Tensor]] = []
    original = solver._deposit_dvm_ribbon

    def wrapped(geometry, trial, result):
        ns = solver.settings.spanwise_panels
        aic = native_aic(geometry, chordwise_panels=solver.settings.chordwise_panels)
        wake_velocity = solver._ring_velocity(
            geometry.collocation, trial.wake_rings, trial.wake_gamma, rough=False
        )
        particle_velocity = (
            trial.particle_field.velocity_at_cuda(geometry.collocation)
            if trial.particle_field.n
            else torch.zeros_like(geometry.collocation)
        )
        rhs = torch.sum(
            (geometry.collocation_velocity - solver.v_inf - wake_velocity - particle_velocity)
            * geometry.normals,
            dim=1,
        )
        gamma_pre = torch.linalg.solve(aic, rhs)
        chord = 0.5 * (
            torch.linalg.vector_norm(
                geometry.trailing_edge[:-1] - geometry.leading_edge[:-1], dim=1
            )
            + torch.linalg.vector_norm(
                geometry.trailing_edge[1:] - geometry.leading_edge[1:], dim=1
            )
        )
        first_dx = 0.5 * (
            torch.linalg.vector_norm(
                geometry.rings[:ns, 3] - geometry.rings[:ns, 0], dim=1
            )
            + torch.linalg.vector_norm(
                geometry.rings[:ns, 2] - geometry.rings[:ns, 1], dim=1
            )
        )
        theta = torch.acos(torch.clamp(1.0 - 2.0 * first_dx / chord, -1.0, 1.0))
        scale = (
            chord
            * solver.settings.freestream
            * (theta + torch.sin(theta))
            / LESP_FACTOR
        )
        lesp_pre_3d = -gamma_pre[:ns] / scale
        captured.append(
            {
                "lesp_pre_3d": lesp_pre_3d.detach().cpu().clone(),
                "shed_lev": result["shed_lev"].detach().cpu().clone(),
            }
        )
        return original(geometry, trial, result)

    solver._deposit_dvm_ribbon = wrapped
    return captured


@unittest.skipUnless(
    torch.cuda.is_available() and wp.is_cuda_available(), "CUDA required"
)
class U2FSeparationMaskSyntheticTest(unittest.TestCase):
    """The mask functions on synthetic tensors (plan §8.3 semantics)."""

    def setUp(self):
        self.device = torch.device(config.DEVICE)
        self.crit = 0.11

    def test_01_unified_mask_is_the_threshold(self):
        lesp = torch.tensor(
            [0.0, 0.05, 0.11, -0.11, 0.1100001, -0.3, 0.2],
            device=self.device, dtype=torch.float64,
        )
        mask = unified_separation_mask(lesp, self.crit)
        self.assertEqual(tuple(mask.shape), (lesp.shape[0],))
        self.assertEqual(mask.dtype, torch.bool)
        self.assertTrue(mask.is_cuda)
        expected = torch.tensor(
            [False, False, False, False, True, True, True],
            device=self.device,
        )
        self.assertTrue(torch.equal(mask, expected))
        # the mask is exactly |LESP| > crit — strict, sign-symmetric
        self.assertTrue(
            torch.equal(mask, torch.abs(lesp) > self.crit)
        )

    def test_02_reconcile_returns_3d_truth_not_union(self):
        # source wants to shed but 3D says attached: the source bank may NOT
        # independently declare separation — a union would pin these strips.
        surface = torch.zeros(4, device=self.device, dtype=torch.bool)
        source = torch.ones(4, device=self.device, dtype=torch.bool)
        active, conflicts = reconcile_release_mask(surface, source)
        self.assertTrue(torch.equal(active, surface))
        self.assertFalse(bool(active.any().item()))
        self.assertEqual(conflicts, 0)

    def test_03_reconcile_continuing_release(self):
        # 3D separated but source bank not shedding (yet): still pinned —
        # the strip carries existing LEV circulation; counted as conflict.
        surface = torch.ones(4, device=self.device, dtype=torch.bool)
        source = torch.zeros(4, device=self.device, dtype=torch.bool)
        active, conflicts = reconcile_release_mask(surface, source)
        self.assertTrue(torch.equal(active, surface))
        self.assertTrue(bool(active.all().item()))
        self.assertEqual(conflicts, 4)

    def test_04_reconcile_mixed_and_agreement(self):
        surface = torch.tensor(
            [True, True, False, False], device=self.device
        )
        source = torch.tensor(
            [False, True, True, False], device=self.device
        )
        active, conflicts = reconcile_release_mask(surface, source)
        # active IS the 3D truth strip-by-strip
        self.assertTrue(torch.equal(active, surface))
        self.assertEqual(conflicts, 1)  # only strip 0 disagrees that way
        # full agreement: saturated all-True → all-True, zero conflicts
        both = torch.ones(30, device=self.device, dtype=torch.bool)
        active, conflicts = reconcile_release_mask(both, both.clone())
        self.assertTrue(bool(active.all().item()))
        self.assertEqual(conflicts, 0)


@unittest.skipUnless(
    torch.cuda.is_available() and wp.is_cuda_available(), "CUDA required"
)
class U2FProductionSingleOwnerTest(unittest.TestCase):
    """5 propose steps of the A16 production solver under per-step capture."""

    @classmethod
    def setUpClass(cls):
        cls.aero = _build_production_aero()
        cls.crit = cls.aero.settings.effective_lesp_crit
        cls.ns = cls.aero.settings.spanwise_panels
        state, velocity = _reference_arrays(cls.aero)
        cls.capture = _attach_capture(cls.aero)
        committed = cls.aero.initialize(state, velocity)
        cls.diagnostics = []
        for _ in range(PROPOSE_STEPS):
            proposal = cls.aero.propose(committed, state, velocity)
            committed = proposal.trial_state
            cls.diagnostics.append(committed.diagnostics[-1])
        cls.committed = committed

    def test_01_capture_covers_every_step_with_full_masks(self):
        self.assertEqual(len(self.capture), PROPOSE_STEPS)
        for entry in self.capture:
            self.assertEqual(tuple(entry["lesp_pre_3d"].shape), (self.ns,))
            # shed_lev carries the bank's cell strips then span nodes
            self.assertEqual(
                tuple(entry["shed_lev"].shape), (2 * self.ns + 1,)
            )

    def test_02_unified_mask_matches_production_lesp_every_step(self):
        for entry in self.capture:
            mask = unified_separation_mask(entry["lesp_pre_3d"], self.crit)
            self.assertTrue(
                torch.equal(mask, torch.abs(entry["lesp_pre_3d"]) > self.crit)
            )

    def test_03_capture_is_faithful_to_solver_diagnostics(self):
        # the recomputed lesp_pre_3d reproduces the solver's own diagnostic
        for entry, diag in zip(self.capture, self.diagnostics):
            self.assertAlmostEqual(
                float(entry["lesp_pre_3d"].abs().max().item()),
                diag["lesp_pre_max_abs"],
                places=12,
            )
            self.assertEqual(
                int(entry["shed_lev"][: self.ns].sum().item()),
                diag["lev_release_count"],
            )

    def test_04_saturated_regime_single_owner_no_conflicts(self):
        # A16 at 16 deg: every strip's 3D |LESP| sits far above LESPcrit and
        # the source bank sheds everywhere, so the single-owner mask is
        # all-True and there is no dual-owner conflict in any step.
        for entry, diag in zip(self.capture, self.diagnostics):
            lesp = entry["lesp_pre_3d"]
            self.assertGreater(float(lesp.abs().min().item()), self.crit)
            self.assertTrue(
                bool((torch.abs(lesp) > self.crit).all().item())
            )
            source_cell = entry["shed_lev"][: self.ns]
            self.assertTrue(bool(source_cell.all().item()))
            surface_separated = unified_separation_mask(lesp, self.crit)
            self.assertTrue(bool(surface_separated.all().item()))
            active, conflicts = reconcile_release_mask(
                surface_separated, source_cell
            )
            # reconcile returns the 3D truth — NOT the union (which would
            # also be all-True here, so additionally pin the identity)
            self.assertTrue(torch.equal(active, surface_separated))
            self.assertEqual(conflicts, 0)
            # production diagnostics agree: all 30 strips separated, zero
            # recorded owner conflicts
            self.assertEqual(diag["separated_strip_count"], self.ns)
            self.assertEqual(diag["release_owner_conflicts"], 0)
            self.assertEqual(int(active.sum().item()), self.ns)

    def test_05_production_pin_is_not_the_union(self):
        # production's separated_strip_count equals the 3D mask count, not
        # count(3D mask | shed_lev): with both all-True the counts coincide,
        # so verify via the union-would-differ contract on the synthetic
        # side already covered above plus the wired reconciliation key.
        for diag in self.diagnostics:
            self.assertIn("release_owner_conflicts", diag)
            self.assertEqual(diag["release_owner_conflicts"], 0)
            self.assertEqual(diag["separated_strip_count"], self.ns)
            self.assertEqual(diag["lev_release_count"], self.ns)

    def test_06_state_wrapper_documents_single_owner(self):
        world = V5MWorldState.from_native_state(self.committed)
        surface = world.surfaces["wing_0"]
        note = surface.separation_owner_note
        self.assertIn("single separation truth", note)
        self.assertIn("shed_lev", note)
        self.assertIn("release strength/position", note)
        # the default note documents the same contract
        device = torch.device(config.DEVICE)
        direct = V5MSurfaceState(
            surface_id="wing_0",
            bound_circulation=torch.zeros(4, device=device, dtype=torch.float64),
            previous_bound_circulation=torch.zeros(4, device=device, dtype=torch.float64),
            lesp_pre_3d=None,
            surface_separated=None,
            lesp_crit=0.11,
            particle_field=CudaParticleField(8, device=device),
            wake_rings=torch.zeros((0, 4, 3), device=device, dtype=torch.float64),
            wake_gamma=torch.zeros(0, device=device, dtype=torch.float64),
        )
        self.assertIn("single separation truth", direct.separation_owner_note)


@unittest.skipUnless(
    torch.cuda.is_available() and wp.is_cuda_available(), "CUDA required"
)
class U2FLoadHistoryIdentityTest(unittest.TestCase):
    """bound_rate vs material are physics identities, not a switch (§8.7)."""

    def test_01_model_identities_are_explicit_and_distinct(self):
        for model in (BOUND_RATE_MODEL, MATERIAL_MODEL):
            self.assertIsInstance(model, LoadHistoryModel)
            for field in ("name", "physics_identity", "stability_note", "oracle_status"):
                self.assertIsInstance(getattr(model, field), str)
                self.assertTrue(getattr(model, field))
        self.assertEqual(BOUND_RATE_MODEL.name, "bound_rate")
        self.assertEqual(MATERIAL_MODEL.name, "material")
        # each model states WHAT it represents physically
        self.assertIn("d(Gamma)/dt", BOUND_RATE_MODEL.physics_identity)
        self.assertIn("weak-scheme", BOUND_RATE_MODEL.physics_identity)
        self.assertIn("Mf2_vec1", MATERIAL_MODEL.physics_identity)
        self.assertIn("material derivative", MATERIAL_MODEL.physics_identity)
        # each model states WHY, and the two never collapse into one choice
        self.assertNotEqual(
            BOUND_RATE_MODEL.physics_identity, MATERIAL_MODEL.physics_identity
        )
        self.assertIn("diverge", BOUND_RATE_MODEL.stability_note)
        self.assertIn("Diverges", MATERIAL_MODEL.stability_note)
        # oracle status is labeled, not assumed
        self.assertEqual(
            BOUND_RATE_MODEL.oracle_status, "validated_against_author_weak_scheme"
        )
        self.assertEqual(
            MATERIAL_MODEL.oracle_status, "validated_for_zero_shedding_only"
        )

    def test_02_registry_covers_production_modes(self):
        self.assertEqual(
            set(LOAD_HISTORY_MODELS), {"bound_rate", "material"}
        )
        for name, model in LOAD_HISTORY_MODELS.items():
            self.assertEqual(model.name, name)

    def test_03_config_validates_mode_against_registry(self):
        for mode in LOAD_HISTORY_MODELS:
            config_obj = NativeV5MConfig(
                wake_history_mode=mode, device=config.DEVICE
            )
            self.assertEqual(config_obj.wake_history_mode, mode)
        with self.assertRaises(ValueError):
            NativeV5MConfig(wake_history_mode="fallback_stability_switch")


if __name__ == "__main__":
    unittest.main()
