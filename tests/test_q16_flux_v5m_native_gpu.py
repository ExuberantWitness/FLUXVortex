from __future__ import annotations

from dataclasses import replace
import inspect
import sys
import unittest

import numpy as np
import torch
import warp as wp

from fluxvortex.warp_fsi import config
from fluxvortex.warp_fsi.q16_flux_v5m_native import (
    NATIVE_V5M_CONTRACT,
    NativeV5MConfig,
    Q16NativeV5MOwner,
    Q16NativeV5MSolver,
    Q16NativeV5MSurface,
    native_aic,
)
from fluxvortex.warp_fsi.q16_flux_v5m_author_loads import (
    material_ring_velocity_derivative_expanded,
)
from forward_flight_benchmarks.yamano2020_q16 import (
    YAMANO_2020_SINGLE_SHEET,
    load_mf1_step1_oracle,
    load_mf2_history_oracle,
    make_yamano2020_q16_model,
)


@unittest.skipUnless(torch.cuda.is_available() and wp.is_cuda_available(), "CUDA required")
class Q16NativeV5MGpuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mesh, _, _ = make_yamano2020_q16_model(
            chordwise_element_count=5,
            spanwise_element_count=3,
        )
        cls.surface = Q16NativeV5MSurface(
            cls.mesh,
            q16_chordwise_elements=5,
            q16_spanwise_elements=3,
            aerodynamic_chordwise_panels=15,
            aerodynamic_spanwise_panels=10,
            device=config.DEVICE,
        )
        cls.solver = Q16NativeV5MSolver(
            cls.surface,
            NativeV5MConfig(
                density=YAMANO_2020_SINGLE_SHEET.fluid_density_kg_m3,
                freestream=YAMANO_2020_SINGLE_SHEET.freestream_m_s,
                aerodynamic_dt=YAMANO_2020_SINGLE_SHEET.aerodynamic_dt_s,
                device=config.DEVICE,
            ),
        )
        cls.state_wp = wp.array(
            np.ascontiguousarray(cls.mesh.reference_state[None, :]),
            dtype=config.DTYPE,
            device=config.DEVICE,
        )
        cls.zero_wp = wp.zeros_like(cls.state_wp)

    def test_production_import_has_no_legacy_runtime(self) -> None:
        self.assertEqual(NATIVE_V5M_CONTRACT, "q16-flux-v5m-native-cuda-float64-v1")
        loaded = tuple(name for name in sys.modules if "pterasoftware" in name.lower())
        self.assertEqual(loaded, ())
        source = inspect.getsource(sys.modules["fluxvortex.warp_fsi.q16_flux_v5m_native"])
        self.assertNotIn("q16_ptera", source.lower())
        self.assertNotIn("author_aero_projection", source.lower())
        self.assertNotIn("Q4", source)
        self.assertNotIn("Q9", source)

    def test_formal_grid_aic_matches_author_oracle(self) -> None:
        geometry = self.surface.evaluate(self.state_wp, self.zero_wp)
        actual = native_aic(geometry, chordwise_panels=15)
        expected = torch.as_tensor(
            np.array(load_mf1_step1_oracle()["aic"], copy=True),
            device="cuda:0",
            dtype=torch.float64,
        )
        relative = torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected)
        self.assertLessEqual(float(relative.item()), 1.0e-6)
        self.assertTrue(bool(torch.all(torch.diagonal(actual) < 0.0).item()))

    def test_author_mf2_history_matches_every_oracle_panel(self) -> None:
        oracle = load_mf2_history_oracle()

        def tensor(step: int, name: str) -> torch.Tensor:
            return torch.as_tensor(
                np.array(oracle[f"step_{step}_{name}"], copy=True),
                device="cuda:0",
                dtype=torch.float64,
            )

        for step in range(1, 5):
            rings = torch.stack(
                tuple(tensor(step, f"r_wake_{corner}") for corner in range(1, 5)),
                dim=1,
            )
            ring_velocity = torch.stack(
                tuple(
                    tensor(step, f"dt_r_wake_{corner}")
                    for corner in range(1, 5)
                ),
                dim=1,
            )
            derivative = material_ring_velocity_derivative_expanded(
                tensor(step, "rc_vec"),
                tensor(step, "dt_rc_vec"),
                rings,
                ring_velocity,
            )
            wake_velocity_rate = torch.sum(
                derivative * tensor(step, "Gamma_wake")[None, :, None], dim=1
            )
            normal_rate = torch.sum(
                wake_velocity_rate * tensor(step, "n_vec_i"), dim=1
            )
            actual = torch.linalg.solve(tensor(step, "A_mat"), -normal_rate)
            torch.testing.assert_close(
                actual,
                tensor(step, "Mf2_vec1"),
                rtol=2.0e-13,
                atol=5.0e-17,
            )

    def test_author_qf_resultant_uses_direct_q16_pressure_map(self) -> None:
        oracle = load_mf2_history_oracle()
        base_geometry = self.surface.evaluate(self.state_wp, self.zero_wp)
        for step in range(1, 5):
            normals = torch.as_tensor(
                np.array(oracle[f"step_{step}_n_vec_i"], copy=True),
                device="cuda:0",
                dtype=torch.float64,
            )
            geometry = replace(base_geometry, normals=normals)
            pressure_map = self.solver.author_load_assembler.pressure_map(geometry)
            pressure = torch.as_tensor(
                np.array(
                    oracle[f"step_{step}_dp_lift1"]
                    + oracle[f"step_{step}_Mf2_vec1"],
                    copy=True,
                ),
                device="cuda:0",
                dtype=torch.float64,
            )
            q16 = pressure_map @ pressure
            actual_force = q16.reshape(-1, 6)[:, :3].sum(dim=0)
            author = torch.as_tensor(
                np.array(oracle[f"step_{step}_Qf_p_global"], copy=True),
                device="cuda:0",
                dtype=torch.float64,
            )
            expected_force = author.reshape(-1, 9)[:, :3].sum(dim=0)
            relative = torch.linalg.vector_norm(
                actual_force - expected_force
            ) / torch.linalg.vector_norm(expected_force)
            self.assertLessEqual(float(relative.item()), 3.0e-3)

    def test_direct_q16_mf1_is_finite_and_opposes_normal_acceleration(self) -> None:
        anchor = self.solver.author_anchor_load(self.state_wp, self.zero_wp)
        matrix = anchor.added_mass.generalized_matrix
        acceleration = torch.zeros(
            self.mesh.dof_count, device="cuda:0", dtype=torch.float64
        )
        reference_rows = torch.as_tensor(
            np.array(self.mesh.reference_rows, copy=True),
            device="cuda:0",
            dtype=torch.float64,
        )
        acceleration[2::6] = reference_rows[:, 0]
        work = acceleration @ matrix @ acceleration
        self.assertTrue(bool(torch.isfinite(matrix).all().item()))
        self.assertLess(float(work.item()), 0.0)
        self.assertGreater(float(torch.linalg.vector_norm(matrix).item()), 0.0)

    def test_trial_is_repeatable_and_does_not_mutate_parent(self) -> None:
        committed = self.solver.initialize(self.state_wp, self.zero_wp)
        before = committed.digest()
        first = self.solver.propose(committed, self.state_wp, self.zero_wp)
        between = committed.digest()
        second = self.solver.propose(committed, self.state_wp, self.zero_wp)
        self.assertEqual(before, between)
        self.assertEqual(before, committed.digest())
        self.assertEqual(first.trial_state.digest(), second.trial_state.digest())
        self.assertEqual(first.trial_state.diagnostics[-1]["lev_release_count"], 0)
        self.assertEqual(first.trial_state.diagnostics[-1]["wake_ring_count"], 10)
        owner = Q16NativeV5MOwner(committed)
        owner.commit(first)
        self.assertEqual(owner.state.step, 1)
        self.assertEqual(owner.state.source_bank.it, 1)

    def test_original_lcrit_releases_only_when_exceeded(self) -> None:
        velocity = np.zeros_like(self.mesh.reference_state)
        velocity.reshape(-1, 6)[:, 2] = 30.0
        velocity_wp = wp.array(
            np.ascontiguousarray(velocity[None, :]),
            dtype=config.DTYPE,
            device=config.DEVICE,
        )
        committed = self.solver.initialize(self.state_wp, velocity_wp)
        proposal = self.solver.propose(committed, self.state_wp, velocity_wp)
        diag = proposal.trial_state.diagnostics[-1]
        self.assertGreater(diag["lesp_pre_max_abs"], 0.11)
        self.assertEqual(diag["lev_release_count"], 10)
        self.assertEqual(diag["separated_strip_count"], 10)
        self.assertGreater(proposal.trial_state.particle_field.n, 0)
        self.assertLessEqual(diag["lesp_pin_max_abs"], 1.0e-6)
        self.assertEqual(diag["kelvin_max_abs"], 0.0)

    def test_direct_q16_load_transfer_closes_force_moment_and_work(self) -> None:
        generator = torch.Generator(device="cuda:0")
        generator.manual_seed(20260823)
        panel_force = torch.randn(
            (150, 3), generator=generator, device="cuda:0", dtype=torch.float64
        )
        generalized = self.solver.load_transfer.map(panel_force)
        generalized_t = wp.to_torch(generalized)[0].reshape(-1, 6)
        rows = wp.to_torch(self.state_wp)[0].reshape(-1, 6)
        geometry = self.surface.evaluate(self.state_wp, self.zero_wp)
        force_error = torch.max(
            torch.abs(torch.sum(generalized_t[:, :3], dim=0) - torch.sum(panel_force, dim=0))
        )
        panel_moment = torch.sum(
            torch.linalg.cross(geometry.collocation, panel_force, dim=1), dim=0
        )
        generalized_moment = torch.sum(
            torch.linalg.cross(rows[:, :3], generalized_t[:, :3], dim=1)
            + torch.linalg.cross(rows[:, 3:], generalized_t[:, 3:], dim=1),
            dim=0,
        )
        moment_error = torch.max(torch.abs(generalized_moment - panel_moment))

        virtual = torch.randn(
            self.mesh.reference_state.shape,
            generator=generator,
            device="cuda:0",
            dtype=torch.float64,
        )
        virtual_wp = wp.from_torch(virtual.unsqueeze(0), dtype=config.DTYPE)
        virtual_geometry = self.surface.evaluate(self.state_wp, virtual_wp)
        aero_work = torch.sum(panel_force * virtual_geometry.collocation_velocity)
        structural_work = torch.sum(wp.to_torch(generalized)[0] * virtual)
        scale = torch.clamp(torch.abs(aero_work), min=1.0)
        self.assertLessEqual(float(force_error.item()), 1.0e-11)
        self.assertLessEqual(float(moment_error.item()), 1.0e-11)
        self.assertLessEqual(float((torch.abs(aero_work - structural_work) / scale).item()), 1.0e-12)

    def test_all_scientific_state_is_cuda_float64(self) -> None:
        state = self.solver.initialize(self.state_wp, self.zero_wp)
        proposal = self.solver.propose(state, self.state_wp, self.zero_wp)
        tensors = (
            proposal.trial_state.gamma_bound,
            proposal.trial_state.wake_rings,
            proposal.trial_state.wake_gamma,
            proposal.trial_state.source_bank.tg,
            proposal.trial_state.particle_field.pos,
            proposal.load.panel_positions,
            proposal.load.panel_forces,
            proposal.load.pressure,
            proposal.author_load.pressure_to_generalized,
            proposal.author_load.dp_lift1,
            proposal.author_load.mf2_history,
            proposal.author_load.constant_pressure,
            proposal.author_load.added_mass.generalized_matrix,
        )
        for value in tensors:
            self.assertEqual(value.device.type, "cuda")
            self.assertIs(value.dtype, torch.float64)
        self.assertTrue(proposal.generalized_force.device.is_cuda)
        self.assertEqual(proposal.generalized_force.dtype, config.DTYPE)


if __name__ == "__main__":
    unittest.main()
